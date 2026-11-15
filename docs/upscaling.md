# DLSS, FSR 3.1, frame generation and Reflex

Optional, off by default. Settings in `mkw.ini` (section `[display]`) or the F1 menu's Screen tab.

| Setting | Values | What it does |
|---|---|---|
| `temporal` | `off`, `dlss`, `fsr3` | NVIDIA DLSS Super Resolution (RTX GPUs; `nvngx_dlss.dll` next to `mkw.exe`) or AMD FSR 3.1 (any GPU; `amd_fidelityfx_loader_vk.dll` and `amd_fidelityfx_upscaler_vk.dll`, or the older `amd_fidelityfx_vk.dll`, next to `mkw.exe`) in the place of the game's own upscale. Live |
| `temporal_quality` | `native`, `quality`, `balanced`, `performance`, `ultra` | The share of the screen's lines the race renders at (100, 67, 58, 50, 33 %). The game renders at most 1440 lines. At the next start |
| `dlss_preset` | `default`, `j`, `k`, `l`, `m` | DLSS's model. Live |
| `temporal_sharpness` | `0`..`100` | FSR 3.1's sharpening. Live |
| `framegen` | `off`, `auto`, `2`, `3`, `4` | Frames shown per game frame (the port's existing setting) |
| `fg_engine` | `flow`, `dlss`, `fsr3` | Who makes those frames: the port's optical flow, DLSS Frame Generation (RTX 40/50; `nvngx_dlssg.dll` next to `mkw.exe`; up to 4x on RTX 50) or FSR 3.1 frame generation (2x; `amd_fidelityfx_framegeneration_vk.dll`). DLSS-G and FSR FG need `temporal` on: they use its depth and motion vectors. Live |
| `reflex` | `off`, `on`, `boost` | NVIDIA Reflex through `VK_NV_low_latency2`. Live |

With `resolution = 2160` (or a window taller than 1440 lines with `temporal` on), the upscaler writes the full output
resolution and the HUD is drawn at it too (`MKW_TEMPORAL_RETARGET=0` turns that off).

## How it works

MKW renders its 3D scene and its screen (HUD) at two resolutions of its own (`GamePerformanceGraphicsConfigGParam` in
the RomFS). With the 3D lines below the screen's, it renders the scene smaller and scales it with one compute dispatch
after SMAA; with them equal, SMAA writes the screen target directly. `runtime/gfx/vk/exec/vk_temporal.c` takes the
place of that dispatch (or of the SMAA blend: DLAA), with:

- jitter: a Halton (2, 3) sub-pixel offset of the viewport of the draws that read the scene's camera block;
- motion vectors: every depth pixel reprojected through the game's own previous view-projection, which its camera
  block keeps (rows 21-31) beside this frame's (`runtime/shaders/tmp_mv.comp`);
- the tonemapped scene before SMAA as colour, the scene's D32 depth.

DLSS and DLSS Frame Generation go through NGX, loaded from the NVIDIA driver at run time (no NVIDIA SDK in the build);
FSR 3.1 through AMD's FidelityFX API DLLs (`runtime/third_party/ffx_sdk`, MIT headers). Frame generation keeps the
port's presenter (`vk_framegen.c`): only the frame synthesis changes. Reflex: `vk_reflex.c` / `vk_latency.c`.

Diagnostics: `MKW_TEMPORAL_DEBUG=mv|nojitter`, `MKW_TEMPORAL_JSIGN`, `MKW_TEMPORAL_TRACE=<frame>`, `MKW_TEMPORAL_KART=0`.
