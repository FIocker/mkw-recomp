// Temporal upscaling (vk_temporal.c): NVIDIA DLSS Super Resolution and AMD FSR 3.1 in the place of MKW's own upscale
// pass, with jitter, camera motion vectors and the game's depth. MKW_TEMPORAL=off|dlss|fsr3 (live),
// MKW_TEMPORAL_QUALITY=native|quality|balanced|performance|ultra (the 3D resolution: applies at the next start).
#pragma once
#include <stdint.h>

#include "gfx.h"
#include "vk.h"
#include "vk_exec.h"

enum { VK_TMP_OFF, VK_TMP_DLSS, VK_TMP_FSR3 };

int vk_tmp_engine(void);   // the engine the setting asks for now (VK_TMP_*)
// The share of the screen's lines the 3D scene renders at for MKW_TEMPORAL_QUALITY, in thousandths; 1000 for native,
// 0 when MKW_TEMPORAL is off (romfs_res.c reads it to write the game's 3D resolution).
int vk_tmp_scene_permille(void);

// Device creation (vk_device.c): what NGX needs, added to the lists; after the device exists, NGX and FFX come up.
void vk_tmp_instance_exts(const char **ext, uint32_t *n, uint32_t max);
void vk_tmp_device_exts(const VkExtensionProperties *have, uint32_t nhave, const char **ext, uint32_t *n, uint32_t max);
void vk_tmp_device_init(void);
void vk_tmp_shutdown(void);   // before the device goes

// The replay (vk_exec_*.c). vk_tmp_draw: each draw once its shaders are known, 0 when the draw is to be skipped (the
// upscaler took its place). vk_tmp_dispatch: each dispatch, 1 when the upscaler replaced it. vk_tmp_viewport: the
// viewport a draw records, with this frame's jitter when the draw is the scene's.
int vk_tmp_draw(VkExec *ex, const GfxShader *vs, const GfxShader *fs);
int vk_tmp_dispatch(VkExec *ex, const GfxShader *cs);
void vk_tmp_viewport(const VkExec *ex, VkViewport *v);
void vk_tmp_present(uint64_t frame);   // the replay thread, at each game present
void vk_tmp_present_size(uint32_t w, uint32_t h);   // the size of the picture that present shows (the game output)
// The screen target at the game output's size (vk_temporal.c): a colour target the frame's upscale replaced
// (vk_exec_rt.c resolve_rt), the viewport and scissor of passes into it, a texture that samples it (vk_exec_view.c)
int vk_tmp_target(VkExecTarget *t);
void vk_tmp_scale_rect(const VkExec *ex, float *x, float *y, float *w, float *h);
void vk_tmp_sampled(VkImage *image, VkImageView *view);

// What the last evaluation left for frame generation (vk_framegen.c): depth and motion vectors at the render size, the
// jitter, and whether the history was reset. 0 when the frame had no temporal upscale.
typedef struct {
    VkImage depth, mv;
    VkImageView depth_view, mv_view;
    VkFormat depth_fmt;
    uint32_t w, h;          // render size
    float jx, jy;           // jitter in render pixels
    int reset;
    float cam[32 * 4];      // the camera block rows 0-31 (view, view-projection, projection, inverse view; 21-31 last frame's)
    double m[16];           // this frame's clip space to the last frame's (row-major, column vectors)
    float near_, far_, fov_y, aspect, frame_ms;
    uint64_t frame;         // the game frame these belong to
} VkTmpFgInputs;
int vk_tmp_fg_inputs(uint64_t frame, VkTmpFgInputs *out);
// DLSS Frame Generation (NGX) for vk_framegen.c: count frames between the last call's picture and bb into out[]; returns
// how many it made, 0 when it cannot (the caller falls back to its own). vk_tmp_ngx_fg_max: the most this GPU makes.
typedef struct { VkImage img; VkImageView view; VkFormat fmt; uint32_t w, h; } VkTmpPic;
int vk_tmp_dlssg(VkCommandBuffer cb, uint64_t frame, const VkTmpPic *bb, const VkTmpPic *out, int count);
int vk_tmp_ngx_fg_max(void);
// FSR 3.1 frame generation (vk_tmp_ffx.cpp): one frame between the last call's picture and bb into out
int vk_tmp_ffx_fg(VkCommandBuffer cb, const VkTmpFgInputs *in, const VkTmpPic *bb, const VkTmpPic *out);

// FFX (vk_tmp_ffx.cpp)
int vk_tmp_ffx_load(void);
typedef struct {
    VkImage image;
    VkFormat format;
    uint32_t w, h;
} VkTmpRes;
// One FSR 3.1 upscale into out; 0 on failure (logged once).
int vk_tmp_ffx_upscale(VkCommandBuffer cb, const VkTmpRes *color, const VkTmpRes *depth, const VkTmpRes *mv,
                       const VkTmpRes *out, float jx, float jy, int reset, float frame_ms, float znear, float zfar,
                       float fov_y, int hdr, float sharpness);
void vk_tmp_ffx_shutdown(void);
// fn(arg) once the GPU is done with the submission being recorded (at once outside a recording)
void vk_tmp_defer(void (*fn)(void *), void *arg);
