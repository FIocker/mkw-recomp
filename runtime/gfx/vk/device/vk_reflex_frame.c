// NVIDIA Reflex for MKW (MKW_REFLEX=off|on|boost, live): the glue between the runtime and vk_reflex.c (the
// VK_NV_low_latency2 calls). One application frame is one game present: its id is the number the back end gives that
// present (vk_backend.c game_frames), the same on the game thread, the replay and the present thread.
//
//   game thread, before the game's frame update (game.c host_frame_update): the Reflex sleep, then SIMULATION_START;
//     after it: SIMULATION_END and RENDERSUBMIT_START (the game records and submits inside its frame update; the
//     replay turns that into Vulkan work as it comes);
//   replay thread, once the frame's last submission is recorded (vk_backend.c present_image): RENDERSUBMIT_END;
//   present thread, around the vkQueuePresentKHR of the game's frame: PRESENT_START and PRESENT_END, with the frame's
//     id as its VkPresentIdKHR. Frames made by frame generation have no own VkPresentIdKHR and use out-of-band present
//     markers with the game frame's id, so they belong to that application frame.
// The swapchain is made with low-latency mode allowed whenever the device has the extension, so the setting changes
// without a new swapchain.
#include <string.h>

#include "cfg.h"
#include "gfx.h"
#include "platform/platform.h"
#include "host.h"
#include "vk.h"
#include "vk_reflex_frame.h"
#include "vk_reflex.h"

static struct {
    VkReflex *r;
    VkReflexSwapchain *sc;
    VkSwapchainKHR sc_handle;
    VkReflexMode mode;         // as last set on the swapchain
    unsigned cfg_seen;
    int want;                  // MKW_REFLEX now
    uint64_t sim_id;           // the frame the game thread is simulating
    uint64_t sleeps, timeouts;
} L = {.cfg_seen = ~0u};

static PlatMutex lk = PLAT_MUTEX_INIT;

void vk_rfx_init(uint32_t spec_version) {
    const VkCtx *c = vk_ctx();
    if (!c || !spec_version) {
        rt_log("reflex: the driver has no VK_NV_low_latency2: NVIDIA Reflex unavailable");
        return;
    }
    L.r = vk_reflex_create(c->device, spec_version);
    vk_reflex_set_external_lock(gfx_swapchain_lock, gfx_swapchain_unlock);
    rt_log("reflex: VK_NV_low_latency2 revision %u %s", spec_version, L.r ? "ready" : "could not be set up");
}

int vk_rfx_available(void) { return L.r && vk_reflex_available(L.r); }

// The menu's note under MKW_REFLEX: 0 none, 1 after a restart (VK_NV_low_latency2 is enabled only with Reflex asked
// for at start, vk_device.c), 2 not available on this PC (no VK_NV_low_latency2: not an NVIDIA driver).
int vk_rfx_menu_state(void) {
    const VkCtx *c = vk_ctx();
    if (!strcmp(cfg_value("MKW_REFLEX"), "off") || !c) return 0;
    if (!c->low_latency2_have) return 2;
    return L.r ? 0 : 1;
}

const void *vk_rfx_swapchain_pnext(VkSwapchainLatencyCreateInfoNV *storage, const void *next) {
    if (!L.r) return next;
    return vk_reflex_swapchain_pnext(L.r, storage, next, 1);
}

static VkReflexMode cfg_mode(void) {
    const char *v = cfg_value("MKW_REFLEX");
    return !strcmp(v, "boost") ? VK_REFLEX_BOOST : !strcmp(v, "on") ? VK_REFLEX_ON : VK_REFLEX_OFF;
}

static void apply_mode_locked(void) {
    if (!L.r || !L.sc) return;
    VkReflexMode m = (VkReflexMode)L.want;
    if (m == L.mode) return;
    VkResult r = vk_reflex_set_mode(L.r, L.sc, m, 0);
    if (r == VK_SUCCESS) L.mode = m;
    rt_log("reflex: low-latency mode %s (%d)", m == VK_REFLEX_BOOST ? "on + boost" : m == VK_REFLEX_ON ? "on" : "off",
           (int)r);
}

void vk_rfx_swapchain_created(VkSwapchainKHR sc) {
    if (!L.r) return;
    plat_mutex_lock(&lk);
    L.sc = vk_reflex_swapchain_attach(L.r, sc);
    L.sc_handle = sc;
    L.mode = VK_REFLEX_OFF;
    L.want = cfg_mode();
    apply_mode_locked();
    plat_mutex_unlock(&lk);
}

void vk_rfx_swapchain_destroying(VkSwapchainKHR sc) {
    if (!L.r || !L.sc || sc != L.sc_handle) return;
    plat_mutex_lock(&lk);
    VkReflexSwapchain *s = L.sc;
    L.sc = NULL, L.sc_handle = VK_NULL_HANDLE;
    plat_mutex_unlock(&lk);
    vk_reflex_swapchain_detach(L.r, s);
}

static int on(void) { return L.r && L.sc && L.mode != VK_REFLEX_OFF; }

void vk_rfx_frame_start(uint64_t id) {
    if (!L.r) return;
    unsigned v = cfg_version();
    if (v != L.cfg_seen) {
        L.cfg_seen = v;
        gfx_swapchain_lock();   // before lk, as create_swapchain (gfx_present.c) takes them
        plat_mutex_lock(&lk);
        L.want = cfg_mode();
        apply_mode_locked();
        plat_mutex_unlock(&lk);
        gfx_swapchain_unlock();
    }
    L.sim_id = id;
    if (!on()) return;
    VkReflexSleepResult s = vk_reflex_sleep(L.r, L.sc);
    L.sleeps++;
    if (s == VK_REFLEX_SLEEP_TIMEOUT) L.timeouts++;
    vk_reflex_marker(L.r, L.sc, VK_LATENCY_MARKER_SIMULATION_START_NV, id);
    vk_reflex_marker(L.r, L.sc, VK_LATENCY_MARKER_INPUT_SAMPLE_NV, id);
    if (L.sleeps % 3600 == 0) rt_log("reflex: %llu sleeps, %llu timed out", (unsigned long long)L.sleeps,
                                     (unsigned long long)L.timeouts);
}

void vk_rfx_frame_end(void) {
    if (!on()) return;
    vk_reflex_marker(L.r, L.sc, VK_LATENCY_MARKER_SIMULATION_END_NV, L.sim_id);
    vk_reflex_marker(L.r, L.sc, VK_LATENCY_MARKER_RENDERSUBMIT_START_NV, L.sim_id);
}

void vk_rfx_submitted(uint64_t id) {
    if (on()) vk_reflex_marker(L.r, L.sc, VK_LATENCY_MARKER_RENDERSUBMIT_END_NV, id);
}

uint64_t vk_rfx_present_start(uint64_t id) {
    if (!on() || !id) return 0;
    vk_reflex_marker(L.r, L.sc, VK_LATENCY_MARKER_PRESENT_START_NV, id);
    return id;
}

void vk_rfx_present_end(uint64_t id) {
    if (on() && id) vk_reflex_marker(L.r, L.sc, VK_LATENCY_MARKER_PRESENT_END_NV, id);
}

uint64_t vk_rfx_oob_present_start(uint64_t id) {
    if (!on() || !id) return 0;
    vk_reflex_marker(L.r, L.sc, VK_LATENCY_MARKER_OUT_OF_BAND_PRESENT_START_NV, id);
    return id;
}

void vk_rfx_oob_present_end(uint64_t id) {
    if (on() && id) vk_reflex_marker(L.r, L.sc, VK_LATENCY_MARKER_OUT_OF_BAND_PRESENT_END_NV, id);
}

void vk_rfx_shutdown(void) {
    if (!L.r) return;
    if (L.sc) vk_reflex_swapchain_detach(L.r, L.sc), L.sc = NULL;
    vk_reflex_destroy(L.r);
    L.r = NULL;
}
