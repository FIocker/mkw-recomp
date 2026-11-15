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
#include <string.h>

#include "ffx_api.h"
#include "ffx_upscale.h"
#include "vk/ffx_api_vk.h"

extern "C" {
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

struct {
    ffxContext ctx = nullptr;
    uint32_t mw = 0, mh = 0, ow = 0, oh = 0;
    int hdr = -1;
    int failed = 0;
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

void destroy_ctx() {
    if (!U.ctx) return;
    vkDeviceWaitIdle(vk_ctx()->device);
    A.destroy(&U.ctx, nullptr);
    U.ctx = nullptr;
}

int context_for(uint32_t mw, uint32_t mh, uint32_t ow, uint32_t oh, int hdr) {
    if (U.ctx && U.ow == ow && U.oh == oh && mw <= U.mw && mh <= U.mh && U.hdr == hdr) return 1;
    destroy_ctx();
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
    ffxReturnCode_t r = A.create(&U.ctx, &d.header, nullptr);
    if (r != FFX_API_RETURN_OK) {
        U.ctx = nullptr;
        if (U.failed++ < 4) rt_log("ffx: upscaling context %ux%u -> %ux%u could not be created (%u)", mw, mh, ow, oh, r);
        return 0;
    }
    U.mw = mw, U.mh = mh, U.ow = ow, U.oh = oh, U.hdr = hdr;
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
    if (!context_for(color->w, color->h, out->w, out->h, hdr)) return 0;
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
    d.viewSpaceToMetersFactor = 0.01f;   // MKW's world is in centimetres (a kart is ~150 units long)
    d.flags = hdr ? 0 : FFX_UPSCALE_FLAG_NON_LINEAR_COLOR_SRGB;
    ffxReturnCode_t r = A.dispatch(&U.ctx, &d.header);
    if (r != FFX_API_RETURN_OK) {
        if (U.failed++ < 4) rt_log("ffx: upscale dispatch failed (%u)", r);
        return 0;
    }
    return 1;
}

extern "C" void vk_tmp_ffx_shutdown(void) {
    if (A.ok) destroy_ctx();
}
