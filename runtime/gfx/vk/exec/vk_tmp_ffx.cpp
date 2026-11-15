// AMD FSR 3.1 upscaling for vk_temporal.c through AMD's FidelityFX API library, loaded at run time from the
// executable's folder: amd_fidelityfx_loader_vk.dll (FidelityFX SDK 2.x, with its upscaler provider
// amd_fidelityfx_upscaler_vk.dll beside it), else the older single amd_fidelityfx_vk.dll. The API's headers are in
// third_party/ffx_sdk (MIT, AMD). One context per render and output size; FFX_UPSCALE_ENABLE_DYNAMIC_RESOLUTION lets
// the game's dynamic resolution change the render size below the context's without a new one.
//
// The inputs are vk_temporal.c's: the tonemapped scene (perceptual colour, so FFX_UPSCALE_FLAG_NON_LINEAR_COLOR_SRGB
// unless the game draws HDR), the D32 depth (0 near, 1 far), and the motion vectors in render pixels from this frame to
// the last. Every image of the replay stays in VK_IMAGE_LAYOUT_GENERAL, which FFX calls the common state.
#ifdef _WIN32
#include <windows.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ffx_api.h"
#include "ffx_upscale.h"
#include "ffx_framegeneration.h"
#include "vk/ffx_api_vk.h"

extern "C" {
#include "cfg.h"
#include "gfx.h"
#include "host.h"
#include "vk.h"
#include "vk_temporal.h"
}

namespace {

struct {
    void *lib = nullptr;
    PfnFfxCreateContext create = nullptr;
    PfnFfxDestroyContext destroy = nullptr;
    PfnFfxConfigure configure = nullptr;
    PfnFfxQuery query = nullptr;
    PfnFfxDispatch dispatch = nullptr;
    int tried = 0, ok = 0, legacy = 0;   // legacy: amd_fidelityfx_vk.dll, which takes the deprecated backend descriptor
} A;

// a few contexts at once, as vk_temporal.c keeps DLSS features: the game's upscale writes the output texture itself in
// some frames, of another size than the screen target
struct Ctx {
    ffxContext ctx = nullptr;
    uint32_t mw = 0, mh = 0, ow = 0, oh = 0;
    int hdr = -1;
    uint64_t used = 0;
};
struct {
    Ctx c[3];
    int cur = 0, failed = 0;
    uint64_t tick = 0;
    ffxContext ctx = nullptr;   // the current one's
} U;

void message(uint32_t type, const wchar_t *msg) {
    char s[512];
    size_t n = 0;
    for (; msg && msg[n] && n < sizeof s - 1; n++) s[n] = msg[n] < 128 ? (char)msg[n] : '?';
    s[n] = 0;
    static int told;
    if (told++ < 40) rt_log("ffx: %s %s", type == FFX_API_MESSAGE_TYPE_ERROR ? "error:" : "warning:", s);
}

uint32_t surface(VkFormat f) {
    switch (f) {
    case VK_FORMAT_B10G11R11_UFLOAT_PACK32: return FFX_API_SURFACE_FORMAT_R11G11B10_FLOAT;
    case VK_FORMAT_R16G16B16A16_SFLOAT: return FFX_API_SURFACE_FORMAT_R16G16B16A16_FLOAT;
    case VK_FORMAT_R16G16_SFLOAT: return FFX_API_SURFACE_FORMAT_R16G16_FLOAT;
    case VK_FORMAT_R32_SFLOAT:   // the depth as plain R32F (vk_temporal.c R.zs)
    case VK_FORMAT_D32_SFLOAT:
    case VK_FORMAT_D32_SFLOAT_S8_UINT: return FFX_API_SURFACE_FORMAT_R32_FLOAT;
    case VK_FORMAT_R8G8B8A8_UNORM: return FFX_API_SURFACE_FORMAT_R8G8B8A8_UNORM;
    case VK_FORMAT_R8G8B8A8_SRGB: return FFX_API_SURFACE_FORMAT_R8G8B8A8_SRGB;
    case VK_FORMAT_A2B10G10R10_UNORM_PACK32: return FFX_API_SURFACE_FORMAT_R10G10B10A2_UNORM;
    default: return FFX_API_SURFACE_FORMAT_UNKNOWN;
    }
}

FfxApiResource res(const VkTmpRes *r, int write) {
    FfxApiResourceDescription d = {};
    d.type = FFX_API_RESOURCE_TYPE_TEXTURE2D;
    d.format = surface(r->format);
    d.width = r->w, d.height = r->h, d.depth = 1, d.mipCount = 1;
    d.flags = FFX_API_RESOURCE_FLAGS_NONE;
    d.usage = write ? FFX_API_RESOURCE_USAGE_UAV : FFX_API_RESOURCE_USAGE_READ_ONLY;
    return ffxApiGetResourceVK((void *)r->image, d, FFX_API_RESOURCE_STATE_COMMON);
}

void ctx_destroy(void *ctx) {   // completion thread
    ffxContext c = (ffxContext)ctx;
    A.destroy(&c, nullptr);
}

void destroy_ctx() {
    vkDeviceWaitIdle(vk_ctx()->device);
    for (auto &c : U.c)
        if (c.ctx) A.destroy(&c.ctx, nullptr), c.ctx = nullptr;
    U.ctx = nullptr;
}

int context_for(uint32_t mw, uint32_t mh, uint32_t ow, uint32_t oh, int hdr, int *changed) {
    U.tick++;
    for (int i = 0; i < 3; i++) {
        Ctx &x = U.c[i];
        if (x.ctx && x.ow == ow && x.oh == oh && mw <= x.mw && mh <= x.mh && x.hdr == hdr) {
            *changed = i != U.cur;
            U.cur = i, x.used = U.tick, U.ctx = x.ctx;
            return 1;
        }
    }
    int pick = 0;
    for (int i = 0; i < 3; i++) {
        if (!U.c[i].ctx) { pick = i; break; }
        if (U.c[i].used < U.c[pick].used) pick = i;
    }
    if (U.c[pick].ctx) {   // the least used one goes once the GPU is done with the submission being recorded
        vk_tmp_defer(ctx_destroy, (void *)U.c[pick].ctx);
        U.c[pick].ctx = nullptr;
    }
    U.cur = pick, *changed = 1;
    const VkCtx *c = vk_ctx();
    ffxCreateContextDescUpscale d = {};
    d.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
    d.flags = FFX_UPSCALE_ENABLE_DYNAMIC_RESOLUTION | (hdr ? FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE : 0);
    d.maxRenderSize = {mw, mh};
    d.maxUpscaleSize = {ow, oh};
    d.fpMessage = message;
    ffxCreateBackendVKDesc be = {};
    ffxCreateBackendVKDesc_DEPRECATED old = {};
    if (A.legacy) {
        old.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK_DEPRECATED;
        old.vkDevice = c->device, old.vkPhysicalDevice = c->phys, old.vkDeviceProcAddr = vkGetDeviceProcAddr;
        d.header.pNext = &old.header;
    } else {
        be.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK;
        be.deviceContext.vkDevice = c->device, be.deviceContext.vkPhysicalDevice = c->phys;
        be.deviceContext.vkDeviceProcAddr = vkGetDeviceProcAddr;
        d.header.pNext = &be.header;
    }
    Ctx &x = U.c[pick];
    ffxReturnCode_t r = A.create(&x.ctx, &d.header, nullptr);
    if (r != FFX_API_RETURN_OK) {
        x.ctx = nullptr, U.ctx = nullptr;
        if (U.failed++ < 4) rt_log("ffx: upscaling context %ux%u -> %ux%u could not be created (%u)", mw, mh, ow, oh, r);
        return 0;
    }
    x.mw = mw, x.mh = mh, x.ow = ow, x.oh = oh, x.hdr = hdr, x.used = U.tick;
    U.ctx = x.ctx;
    rt_log("ffx: FSR upscaling context %ux%u -> %ux%u%s", mw, mh, ow, oh, hdr ? ", HDR" : "");
    return 1;
}

}  // namespace

extern "C" int vk_tmp_ffx_load(void) {
    if (A.tried) return A.ok;
    A.tried = 1;
#ifdef _WIN32
    char path[MAX_PATH];
    DWORD n = GetModuleFileNameA(NULL, path, MAX_PATH);
    while (n && path[n - 1] != '\\') n--;
    static const char *const names[] = {"amd_fidelityfx_loader_vk.dll", "amd_fidelityfx_vk.dll"};
    HMODULE m = nullptr;
    for (int i = 0; i < 2 && !m; i++) {
        snprintf(path + n, MAX_PATH - n, "%s", names[i]);
        m = LoadLibraryA(path);
        A.legacy = i == 1;
    }
    if (!m) {
        rt_log("ffx: neither amd_fidelityfx_loader_vk.dll nor amd_fidelityfx_vk.dll next to mkw.exe: FSR 3.1 unavailable");
        return 0;
    }
    A.lib = m;
    A.create = (PfnFfxCreateContext)(void *)GetProcAddress(m, "ffxCreateContext");
    A.destroy = (PfnFfxDestroyContext)(void *)GetProcAddress(m, "ffxDestroyContext");
    A.configure = (PfnFfxConfigure)(void *)GetProcAddress(m, "ffxConfigure");
    A.query = (PfnFfxQuery)(void *)GetProcAddress(m, "ffxQuery");
    A.dispatch = (PfnFfxDispatch)(void *)GetProcAddress(m, "ffxDispatch");
    A.ok = A.create && A.destroy && A.dispatch;
    rt_log("ffx: %s %s", path, A.ok ? "loaded" : "lacks the API entry points: FSR 3.1 unavailable");
#else
    rt_log("ffx: FSR 3.1 is only wired on Windows in this build");
#endif
    return A.ok;
}

extern "C" int vk_tmp_ffx_upscale(VkCommandBuffer cb, const VkTmpRes *color, const VkTmpRes *depth, const VkTmpRes *mv,
                                  const VkTmpRes *out, float jx, float jy, int reset, float frame_ms, float znear,
                                  float zfar, float fov_y, int hdr, float sharpness) {
    if (!A.ok) return 0;
    int changed = 0;
    if (!context_for(color->w, color->h, out->w, out->h, hdr, &changed)) return 0;
    if (changed) reset = 1;
    ffxDispatchDescUpscale d = {};
    d.header.type = FFX_API_DISPATCH_DESC_TYPE_UPSCALE;
    d.commandList = (void *)cb;
    d.color = res(color, 0);
    d.depth = res(depth, 0);
    d.motionVectors = res(mv, 0);
    d.output = res(out, 1);
    d.jitterOffset = {jx, jy};
    d.motionVectorScale = {1.0f, 1.0f};
    d.renderSize = {color->w, color->h};
    d.upscaleSize = {out->w, out->h};
    d.enableSharpening = sharpness > 0;
    d.sharpness = sharpness > 1 ? 1.f : sharpness;
    d.frameTimeDelta = frame_ms;
    d.preExposure = 1.0f;
    d.reset = reset != 0;
    d.cameraNear = znear;
    d.cameraFar = zfar;
    d.cameraFovAngleVertical = fov_y;
    // Same metre-scale camera/depth units as frame generation below (Mario's body spans about 1.74 units).
    d.viewSpaceToMetersFactor = 1.0f;
    d.flags = hdr ? 0 : FFX_UPSCALE_FLAG_NON_LINEAR_COLOR_SRGB;
    ffxReturnCode_t r = A.dispatch(&U.c[U.cur].ctx, &d.header);
    if (r != FFX_API_RETURN_OK) {
        if (U.failed++ < 4) rt_log("ffx: upscale dispatch failed (%u)", r);
        return 0;
    }
    return 1;
}

// ---- frame generation: FSR 3.1's interpolation without its swapchain (FFX_FRAMEGENERATION_FLAG_NO_SWAPCHAIN_CONTEXT_NOTIFY):
// vk_framegen.c's present thread presents the frame like its own made ones. Each game frame: the prepare pass (depth,
// motion vectors, camera) and the generation, both into the present thread's command buffer.
namespace {
struct {
    ffxContext ctx = nullptr;
    uint32_t w = 0, h = 0, rw = 0, rh = 0;
    VkFormat fmt = VK_FORMAT_UNDEFINED;
    uint64_t id = 0, last_frame = 0, evals = 0;
    int failed = 0;
} FG;

void fg_destroy() {
    if (!FG.ctx) return;
    vk_queue_lock();   // the present thread, as vk_framegen.c ensure_images
    vkDeviceWaitIdle(vk_ctx()->device);
    vk_queue_unlock();
    A.destroy(&FG.ctx, nullptr);
    FG.ctx = nullptr;
}

FfxApiResource pic(const VkTmpPic *p, int write) {
    VkTmpRes r = {p->img, p->fmt, p->w, p->h};
    return res(&r, write);
}
}  // namespace

extern "C" int vk_tmp_ffx_fg(VkCommandBuffer cb, const VkTmpFgInputs *in, const VkTmpPic *bb, const VkTmpPic *hl,
                             const VkTmpPic *out) {
    const char *full_mv = cfg_get("MKW_FG_DISPLAY_MV");
    int display_mv = full_mv && strcmp(full_mv, "0") && cfg_get("MKW_FG_NVOF") && strcmp(cfg_get("MKW_FG_NVOF"), "0");
    if (!vk_tmp_ffx_load() || FG.failed) return 0;
    const VkCtx *c = vk_ctx();
    if (FG.ctx && (FG.w != bb->w || FG.h != bb->h || FG.fmt != bb->fmt || in->w > FG.rw || in->h > FG.rh)) fg_destroy();
    if (!FG.ctx) {
        ffxCreateContextDescFrameGeneration d = {};
        d.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION;
        ffxCreateContextDescFrameGenerationVersion api = {};
        api.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION_VERSION;
        api.version = FFX_FRAMEGENERATION_VERSION;
        int hdr = bb->fmt != VK_FORMAT_R8G8B8A8_UNORM && bb->fmt != VK_FORMAT_R8G8B8A8_SRGB;
        d.flags = hdr ? FFX_FRAMEGENERATION_ENABLE_HIGH_DYNAMIC_RANGE : 0;
        if (display_mv) d.flags |= FFX_FRAMEGENERATION_ENABLE_DISPLAY_RESOLUTION_MOTION_VECTORS;
        d.displaySize = {bb->w, bb->h};
        d.maxRenderSize = {bb->w, bb->h};
        d.backBufferFormat = surface(bb->fmt);
        ffxCreateBackendVKDesc be = {};
        ffxCreateBackendVKDesc_DEPRECATED old = {};
        if (A.legacy) {
            old.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK_DEPRECATED;
            old.vkDevice = c->device, old.vkPhysicalDevice = c->phys, old.vkDeviceProcAddr = vkGetDeviceProcAddr;
            api.header.pNext = &old.header;
        } else {
            be.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK;
            be.deviceContext.vkDevice = c->device, be.deviceContext.vkPhysicalDevice = c->phys;
            be.deviceContext.vkDeviceProcAddr = vkGetDeviceProcAddr;
            api.header.pNext = &be.header;
        }
        // SDK 2.1+ requires the application API version in the creation chain for new frame-generation APIs.
        // Keep the backend chained after it so PrepareV2's camera data is interpreted under the declared contract.
        d.header.pNext = &api.header;
        // under the queue lock, and finished before it is let go: FFX's backend may submit its own setup work to the
        // device's queue while it makes the context (two runs faulted on a write to address 0 at the first frame)
        vk_queue_lock();
        ffxReturnCode_t r = A.create(&FG.ctx, &d.header, nullptr);
        vkDeviceWaitIdle(c->device);
        vk_queue_unlock();
        if (r != FFX_API_RETURN_OK) {
            FG.ctx = nullptr, FG.failed = 1;
            rt_log("ffx: frame generation context %ux%u could not be created (%u): the port's own frame generation",
                   bb->w, bb->h, r);
            return 0;
        }
        FG.w = bb->w, FG.h = bb->h, FG.rw = bb->w, FG.rh = bb->h, FG.fmt = bb->fmt, FG.id = 0;
        rt_log("ffx: FSR 3.1 frame generation context %ux%u", bb->w, bb->h);
        if (A.query) {
            ffxQueryGetProviderVersion version = {};
            version.header.type = FFX_API_QUERY_DESC_TYPE_GET_PROVIDER_VERSION;
            if (A.query(&FG.ctx, &version.header) == FFX_API_RETURN_OK)
                rt_log("ffx: selected frame-generation provider %s (id %llu)",
                       version.versionName ? version.versionName : "unnamed", (unsigned long long)version.versionId);
        }
    }
    int reset = in->reset || (FG.last_frame && in->frame != FG.last_frame + 1);
    FG.id++;
    ffxConfigureDescFrameGeneration cf = {};
    cf.header.type = FFX_API_CONFIGURE_DESC_TYPE_FRAMEGENERATION;
    cf.swapChain = nullptr;
    cf.frameGenerationEnabled = true;
    cf.allowAsyncWorkloads = false;
    cf.flags = FFX_FRAMEGENERATION_FLAG_NO_SWAPCHAIN_CONTEXT_NOTIFY;
    // Opt-in SDK diagnostics expose its motion/occlusion decisions in the generated picture. Keep NO_SWAPCHAIN:
    // this port owns presentation even when inspecting the provider's debug view.
    const char *debug = cfg_get("MKW_FSR_FG_DEBUG");
    if (debug && strcmp(debug, "0")) cf.flags |= FFX_FRAMEGENERATION_FLAG_DRAW_DEBUG_VIEW;
    cf.generationRect = {0, 0, (int32_t)bb->w, (int32_t)bb->h};
    cf.frameID = FG.id;
    // the picture without the HUD (vk_tmp_fg_hudless; bb's format and size): FSR 3.1 takes the HUD from the
    // difference and keeps it out of the interpolation
    if (hl) cf.HUDLessColor = pic(hl, 0);
    ffxReturnCode_t r = A.configure(&FG.ctx, &cf.header);
    if (r != FFX_API_RETURN_OK) {
        FG.failed = 1;
        rt_log("ffx: frame generation configure failed (%u): the port's own frame generation", r);
        return 0;
    }
    ffxDispatchDescFrameGenerationPrepareV2 pr = {};
    pr.header.type = FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION_PREPARE_V2;
    pr.frameID = FG.id;
    pr.commandList = (void *)cb;
    pr.renderSize = {in->w, in->h};
    pr.jitterOffset = {in->jx, in->jy};
    pr.motionVectorScale = {1.0f, 1.0f};
    pr.frameTimeDelta = in->frame_ms;
    pr.reset = reset != 0;
    pr.cameraNear = in->near_;
    pr.cameraFar = in->far_;
    pr.cameraFovAngleVertical = in->fov_y;
    // Game assets use metre-scale positions (Mario's body bounds are about 1.74 units, unit bone scale), matching
    // the race depth's ~4.2-unit camera-to-driver distance. A centimetre conversion shrinks the SDK's scene by 100x.
    pr.viewSpaceToMetersFactor = 1.0f;
    // Diagnostic calibration override for comparing the former centimetre assumption against the measured units.
    const char *meters = cfg_get("MKW_FG_METERS");
    if (meters) {
        float scale = strtof(meters, nullptr);
        if (scale >= 0.0001f && scale <= 100.f) pr.viewSpaceToMetersFactor = scale;
    }
    VkTmpRes rz = {in->depth, VK_FORMAT_R32_SFLOAT, in->w, in->h},
             rm = {in->mv, VK_FORMAT_R16G16_SFLOAT, display_mv ? bb->w : in->w, display_mv ? bb->h : in->h};
    pr.depth = res(&rz, 0);
    pr.motionVectors = res(&rm, 0);
    const float *v = in->cam;
    for (int i = 0; i < 3; i++) {
        pr.cameraRight[i] = v[i], pr.cameraUp[i] = v[4 + i], pr.cameraForward[i] = -v[8 + i];
        pr.cameraPosition[i] = -(v[i] * v[3] + v[4 + i] * v[7] + v[8 + i] * v[11]);
    }
    r = A.dispatch(&FG.ctx, &pr.header);
    if (r != FFX_API_RETURN_OK) {
        if (FG.evals < 4) rt_log("ffx: frame generation prepare failed (%u)", r);
        return 0;
    }
    ffxDispatchDescFrameGeneration g = {};
    g.header.type = FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION;
    g.commandList = (void *)cb;
    g.presentColor = pic(bb, 0);
    g.outputs[0] = pic(out, 1);
    g.numGeneratedFrames = 1;
    g.reset = reset != 0;
    int hdr = bb->fmt != VK_FORMAT_R8G8B8A8_UNORM && bb->fmt != VK_FORMAT_R8G8B8A8_SRGB;
    uint16_t peak = 0, white = 0;
    if (hdr) gfx_hdr_output(&peak, &white);
    g.backbufferTransferFunction = !hdr ? FFX_API_BACKBUFFER_TRANSFER_FUNCTION_SRGB
                                 : bb->fmt == VK_FORMAT_R16G16B16A16_SFLOAT ? FFX_API_BACKBUFFER_TRANSFER_FUNCTION_SCRGB
                                                                          : FFX_API_BACKBUFFER_TRANSFER_FUNCTION_PQ;
    g.minMaxLuminance[0] = hdr ? 0.001f : 0.f, g.minMaxLuminance[1] = hdr ? (peak ? (float)peak : 1000.f) : 1.f;
    g.generationRect = {0, 0, (int32_t)bb->w, (int32_t)bb->h};
    g.frameID = FG.id;
    r = A.dispatch(&FG.ctx, &g.header);
    if (r != FFX_API_RETURN_OK) {
        if (FG.evals < 4) rt_log("ffx: frame generation dispatch failed (%u)", r);
        return 0;
    }
    FG.last_frame = in->frame;
    if (++FG.evals == 1 || FG.evals % 3600 == 0)
        rt_log("ffx: FSR 3.1 frame generation: %llu game frames", (unsigned long long)FG.evals);
    return 1;
}

extern "C" void vk_tmp_ffx_shutdown(void) {
    if (A.ok) destroy_ctx(), fg_destroy();
}
