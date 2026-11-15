// --probe-fsr-fg: asset-free FSR integration check with known geometry and shading motion.
// A textured plane translates beneath a camera-attached rectangle and its fixed shadow. Geometry vectors belong
// to the plane under the shadow; image vectors do not. Compare both with MKW_FSR_PROBE_MV=geometry|image|zero.
// Explicit CPU waits are intentional here: this tests the SDK contract, never production frame pacing.
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cfg.h"
#include "host.h"
#include "vk.h"
#include "vk_rec.h"
#include "vk_temporal.h"

#define W 256
#define H 128
#define BYTES (W * H * 4)
#define CHECK(x) do { if (!(x)) { printf("probe-fsr-fg: FAILED line %d: %s\n", __LINE__, #x); return 1; } } while (0)

static float motion_x = -4, motion_y = 2;
static int opaque_shadow;
static int still_images;
static uint16_t half(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof bits);
    if (!(bits & 0x7fffffff)) return (uint16_t)(bits >> 16);
    // This probe accepts integer displacements in [-128,128], all exactly representable in binary16.
    return (uint16_t)((bits >> 16 & 0x8000) | (((bits >> 23 & 255) - 112) << 10) | (bits >> 13 & 1023));
}
static int body(int x, int y) { return x >= 104 && x < 152 && y >= 28 && y < 78; }
static int shadow(int x, int y) { return x >= 88 && x < 168 && y >= 76 && y < 98; }
static uint8_t channel(int x, int y, double t, int k) {
    if (still_images) t = 0;
    if (body(x, y)) return (uint8_t)(60 + ((x * (k + 3) + y * (k + 5)) % 160));
    if (opaque_shadow && shadow(x,y)) return 32;
    double xx = x + motion_x * t, yy = y + motion_y * t;
    double p = 128 + 35 * sin(xx * .13 + yy * .031 + k) + 33 * cos(yy * .19 + xx * .017 + k * .7)
                   + 28 * sin(xx * .071 - yy * .11);
    return (uint8_t)(p * (shadow(x, y) ? .25 : 1.0));
}
static void barrier(VkCommandBuffer cb) {
    VkMemoryBarrier2 b = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT};
    VkDependencyInfo d = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &b};
    vkCmdPipelineBarrier2(cb, &d);
}
static int finish(VkCtx *c, VkCommandBuffer cb, VkFence fence) {
    if (vkEndCommandBuffer(cb) != VK_SUCCESS) return 0;
    VkCommandBufferSubmitInfo ci = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, .commandBuffer = cb};
    VkSubmitInfo2 si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2, .commandBufferInfoCount = 1, .pCommandBufferInfos = &ci};
    vk_queue_lock();
    VkResult r = vkQueueSubmit2(c->gfx_queue, 1, &si, fence);
    vk_queue_unlock();
    return r == VK_SUCCESS && vkWaitForFences(c->device, 1, &fence, VK_TRUE, 10000000000ull) == VK_SUCCESS;
}

int probe_fsr_fg(void) {
    CHECK(vk_init());
    VkCtx *c = vk_ctx();
    CHECK(vk_tmp_ffx_load());
    int recording = rec_suspend();
    const char *mode = cfg_get("MKW_FSR_PROBE_MV");
    if (!mode) mode = "geometry";
    CHECK(!strcmp(mode, "geometry") || !strcmp(mode, "image") || !strcmp(mode, "zero"));
    const char *opaque = cfg_get("MKW_FSR_PROBE_OPAQUE");
    opaque_shadow = opaque && strcmp(opaque,"0");
    const char *still = cfg_get("MKW_FSR_PROBE_STILL");
    still_images = still && strcmp(still,"0");
    const char *shift = cfg_get("MKW_FSR_PROBE_SHIFT");
    if (shift) {
        CHECK(sscanf(shift,"%f,%f",&motion_x,&motion_y) == 2);
        CHECK(isfinite(motion_x) && isfinite(motion_y) && fabsf(motion_x) <= 128 && fabsf(motion_y) <= 128 &&
              motion_x == roundf(motion_x) && motion_y == roundf(motion_y));
    }
    VkBuffer staging; VmaAllocation memory; VmaAllocationInfo map;
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = BYTES * 4,
        .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT};
    VmaAllocationCreateInfo host = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
        .flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT};
    CHECK(vmaCreateBuffer(c->vma, &bi, &host, &staging, &memory, &map) == VK_SUCCESS);
    uint8_t *bytes = map.pMappedData;
    VkFormat formats[4] = {VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R32_SFLOAT, VK_FORMAT_R16G16_SFLOAT,
                          VK_FORMAT_R8G8B8A8_UNORM};
    VkImage images[4]; VkImageView views[4]; VmaAllocation allocations[4];
    for (int i = 0; i < 4; i++) {
        VkImageCreateInfo im = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D,
            .format = formats[i], .extent = {W,H,1}, .mipLevels = 1, .arrayLayers = 1,
            .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
            .usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                     VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT};
        VmaAllocationCreateInfo device = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
        CHECK(vmaCreateImage(c->vma, &im, &device, &images[i], &allocations[i], NULL) == VK_SUCCESS);
        VkImageViewCreateInfo v = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = images[i],
            .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = formats[i],
            .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1}};
        CHECK(vkCreateImageView(c->device, &v, NULL, &views[i]) == VK_SUCCESS);
    }
    VkCommandPool pool;
    VkCommandPoolCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, .queueFamilyIndex = c->gfx_family};
    CHECK(vkCreateCommandPool(c->device, &pi, NULL, &pool) == VK_SUCCESS);
    VkCommandBuffer cb;
    VkCommandBufferAllocateInfo ca = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
    CHECK(vkAllocateCommandBuffers(c->device, &ca, &cb) == VK_SUCCESS);
    VkFence fence;
    VkFenceCreateInfo fi = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    CHECK(vkCreateFence(c->device, &fi, NULL, &fence) == VK_SUCCESS);
    for (int frame = 0; frame <= 20; frame++) {
        for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
            int p = y * W + x;
            for (int k = 0; k < 3; k++) bytes[p * 4 + k] = channel(x, y, frame, k);
            bytes[p * 4 + 3] = 255;
            // Finite perspective near=1, far=100, body at 2 units, road at 4.
            ((float *)(bytes + BYTES))[p] = (100.f - 100.f / (body(x,y) ? 2.f : 4.f)) / 99.f;
            int still = body(x,y) || !strcmp(mode, "zero") || (!strcmp(mode,"image") && shadow(x,y));
            uint16_t *mv = (uint16_t *)(bytes + BYTES * 2) + p * 2;
            mv[0] = still ? 0 : half(motion_x);
            mv[1] = still ? 0 : half(motion_y);
        }
        CHECK(vmaFlushAllocation(c->vma, memory, 0, BYTES * 3) == VK_SUCCESS);
        if (frame) CHECK(vkResetFences(c->device, 1, &fence) == VK_SUCCESS);
        CHECK(vkResetCommandBuffer(cb, 0) == VK_SUCCESS);
        VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
        CHECK(vkBeginCommandBuffer(cb, &begin) == VK_SUCCESS);
        if (!frame) for (int i = 0; i < 4; i++) {
            VkImageMemoryBarrier2 b = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = images[i], .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1}};
            VkDependencyInfo d = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 1,
                .pImageMemoryBarriers = &b};
            vkCmdPipelineBarrier2(cb, &d);
        }
        barrier(cb);
        for (int i = 0; i < 3; i++) {
            VkBufferImageCopy cp = {.bufferOffset = (VkDeviceSize)i * BYTES,
                .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT,0,0,1}, .imageExtent = {W,H,1}};
            vkCmdCopyBufferToImage(cb, staging, images[i], VK_IMAGE_LAYOUT_GENERAL, 1, &cp);
        }
        barrier(cb);
        VkTmpFgInputs in = {.depth = images[1], .mv = images[2], .depth_view = views[1], .mv_view = views[2],
            .depth_fmt = VK_FORMAT_R32_SFLOAT, .w = W, .h = H, .reset = !frame,
            .near_ = 1, .far_ = 100, .fov_y = 1.04719755f, .aspect = 2, .frame_ms = 1000.f/60, .frame = frame+1};
        in.cam[0] = in.cam[5] = in.cam[10] = 1;
        VkTmpPic color = {images[0],views[0],formats[0],W,H}, out = {images[3],views[3],formats[3],W,H};
        CHECK(vk_tmp_ffx_fg(cb, &in, &color, NULL, &out));
        // Experimental provider pipelines must pass shader/descriptor validation before any of their work is submitted.
        CHECK(!vk_validation_errors_total());
        barrier(cb);
        VkBufferImageCopy cp = {.bufferOffset = BYTES * 3,
            .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT,0,0,1}, .imageExtent = {W,H,1}};
        vkCmdCopyImageToBuffer(cb, images[3], VK_IMAGE_LAYOUT_GENERAL, staging, 1, &cp);
        VkMemoryBarrier2 read = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
            .srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT, .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT, .dstAccessMask = VK_ACCESS_2_HOST_READ_BIT};
        VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1,
            .pMemoryBarriers = &read};
        vkCmdPipelineBarrier2(cb, &di);
        CHECK(finish(c, cb, fence));
    }
    CHECK(vmaInvalidateAllocation(c->vma, memory, 0, VK_WHOLE_SIZE) == VK_SUCCESS);
    double error[3] = {0}; int count[3] = {0}, bad[3] = {0};
    for (int y = 16; y < H-16; y++) for (int x = 16; x < W-16; x++) {
        int group = body(x,y) ? 0 : shadow(x,y) ? 1 : 2;
        int worst = 0;
        for (int k = 0; k < 3; k++) {
            int e = abs((int)bytes[BYTES*3 + (y*W+x)*4+k] - channel(x,y,19.5,k));
            error[group] += e;
            if (e > worst) worst = e;
        }
        count[group]++; bad[group] += worst > 24;
    }
    printf("probe-fsr-fg: %s vectors (%.0f,%.0f), %s shadow, analytical midpoint after 20 frames; validation errors %ld\n",
        mode,motion_x,motion_y,opaque_shadow?"opaque":"textured",vk_validation_errors_total());
    if (still_images) printf("probe-fsr-fg: identical endpoint images; injected nonzero world vectors intentionally inconsistent\n");
    const char *names[3] = {"body", "shadow", "road"};
    for (int i = 0; i < 3; i++) printf("probe-fsr-fg: %s mean RGB error %.3f /255, >24 max-RGB %d/%d\n",
        names[i], error[i]/(3*count[i]), bad[i], count[i]);
    const char *path = cfg_get("MKW_FSR_PROBE_OUT");
    if (path) {
        FILE *f = fopen(path, "wb");
        CHECK(f);
        fprintf(f,"P6\n%d %d\n255\n",W,H);
        for (int p = 0; p < W*H; p++) fwrite(bytes+BYTES*3+p*4,1,3,f);
        CHECK(!fclose(f));
    }
    int ok = !vk_validation_errors_total() && error[2]/(3*count[2]) < 8;
    const char *limit = cfg_get("MKW_FSR_PROBE_SHADOW_LIMIT");
    if (limit) {
        float maximum = strtof(limit,NULL);
        CHECK(isfinite(maximum) && maximum > 0);
        int quality = error[1]/(3*count[1]) <= maximum && error[0]/(3*count[0]) <= 2;
        printf("probe-fsr-fg: analytical shadow quality gate <= %.3f /255, body <=2: %s\n",maximum,quality?"passed":"FAILED");
        ok = ok && quality;
    }
    vk_tmp_ffx_shutdown();
    vkDestroyFence(c->device,fence,NULL); vkDestroyCommandPool(c->device,pool,NULL);
    for (int i = 0; i < 4; i++) { vkDestroyImageView(c->device,views[i],NULL); vmaDestroyImage(c->vma,images[i],allocations[i]); }
    vmaDestroyBuffer(c->vma,staging,memory);
    rec_resume(recording);
    printf("probe-fsr-fg: %s (quality metrics above remain separate from this integration check)\n",ok?"passed":"FAILED");
    return ok ? 0 : 1;
}
