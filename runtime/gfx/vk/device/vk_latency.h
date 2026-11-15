// NVIDIA Reflex for MKW (vk_latency.c): the runtime's side of vk_reflex.c. MKW_REFLEX=off|on|boost, live.
#pragma once
#include <stdint.h>
#include <vulkan/vulkan.h>

void vk_lat_init(uint32_t spec_version);   // after the device, with VK_NV_low_latency2's enabled revision (0: none)
int vk_lat_available(void);
const void *vk_lat_swapchain_pnext(VkSwapchainLatencyCreateInfoNV *storage, const void *next);
void vk_lat_swapchain_created(VkSwapchainKHR sc);
void vk_lat_swapchain_destroying(VkSwapchainKHR sc);   // before vkDestroySwapchainKHR
void vk_lat_frame_start(uint64_t id);   // game thread, before the frame update: the sleep and the simulation's start
void vk_lat_frame_end(void);            // game thread, after it
void vk_lat_submitted(uint64_t id);     // replay thread: the frame's work is recorded
uint64_t vk_lat_present_start(uint64_t id);   // present thread: the VkPresentIdKHR to present the game frame with, 0 none
void vk_lat_present_end(uint64_t id);
void vk_lat_shutdown(void);
uint64_t vk_be_next_frame_id(void);     // vk_backend.c: the id of the frame the game thread is about to simulate
