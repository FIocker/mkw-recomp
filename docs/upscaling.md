# DLSS, FSR 3.1, frame generation and Reflex

Optional, off by default. Settings in `mkw.ini` (section `[display]`) or the F1 menu's Screen tab.

| Setting | Values | What it does |
|---|---|---|
| `temporal` | `off`, `dlss`, `fsr3` | NVIDIA DLSS Super Resolution (RTX GPUs; `nvngx_dlss.dll` next to `mkw.exe`) or AMD FSR 3.1 (any GPU; `amd_fidelityfx_loader_vk.dll` and `amd_fidelityfx_upscaler_vk.dll`, or the older `amd_fidelityfx_vk.dll`, next to `mkw.exe`) in the place of the game's own upscale. Live |
| `temporal_quality` | `native`, `quality`, `balanced`, `performance`, `ultra` | The share of the screen's lines the race renders at (100, 67, 58, 50, 33 %). The game renders at most 1440 lines. At the next start |
| `dlss_preset` | `default`, `j`, `k`, `l`, `m` | DLSS's model. Live |
| `temporal_sharpness` | `0`..`100` | FSR 3.1's sharpening. Live |
| `cas` | `off`, `on` | AMD's contrast adaptive sharpening (CAS) of the race after DLSS, DLAA or FSR 3.1, before the HUD. Needs `temporal` on. Live |
| `cas_sharpness` | `0`..`100` | CAS's strength. Live |
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

The player's kart moves with the camera, so camera motion is wrong for it: near pixels in the kart's part of the screen
that match the last frame's picture better standing still than moved with the camera get no motion (a circle around
the kart's place on the first frame; `MKW_TEMPORAL_KART_CIRCLE=1` keeps the circle).

Known: with the NVIDIA App's DLSS Frame Generation override on (NVIDIA App > Graphics > DLSS Override - Frame
Generation), NGX loads the App's model with its render preset 3 and the made frames come out as an even blend of the two
game frames, whatever depth, motion vectors or camera they are given (measured with `MKW_FG_DUMP`). With the override off,
the `nvngx_dlssg.dll` next to `mkw.exe` interpolates properly. `MKW_NGX_LOG=1` logs which model was loaded.

Diagnostics: `MKW_TEMPORAL_DEBUG=mv|nojitter`, `MKW_TEMPORAL_JSIGN`, `MKW_TEMPORAL_TRACE=<frame>`, `MKW_TEMPORAL_KART=0`,
`MKW_NGX_LOG=1|2` (NGX's own log lines in ours), `MKW_DLSSG_EXP=<mask>` (vk_temporal.c vk_tmp_dlssg), `MKW_FG_DUMP`.

## Frame-generation diagnostics under development

`--probe-fsr-fg` runs the real FSR provider without game assets or a window, using an analytical moving plane and a
fixed foreground object and shadow. `MKW_FSR_PROBE_MV=geometry|image|zero` selects the known motion input;
`MKW_FSR_PROBE_SHIFT=0,-64` sets integer pixel displacement, and `MKW_FSR_PROBE_OPAQUE=1` makes the shadow a constant
opaque patch with unambiguous zero image motion. `MKW_FSR_PROBE_OUT=<file.ppm>` writes the generated midpoint.
`MKW_FSR_PROBE_SHADOW_LIMIT=2` enables a quality gate against the analytical midpoint, in addition to the integration
check. On provider 3.1.6, the opaque image-motion case passes at `(0,-8)` but fails at `(0,-64)`: the body remains exact,
while shadow mean error rises from 0.225 to 41.429 levels of 255. A successful SDK dispatch alone is not quality acceptance.

Experimental motion controls default off: `MKW_FG_NVOF` uses the Vulkan hardware optical-flow queue on supported NVIDIA
devices; `MKW_FG_NVOF_GRID=1|2|4` changes its vector grid (default 4). `MKW_FG_NVOF_GEOMETRY`, `MKW_FG_NVOF_REFINE`, and
`MKW_FG_NVOF_CAMERA` test depth classification, endpoint correspondence, and camera agreement respectively. Race captures
still show artifacts with each of these controls; none is an accepted quality fix. `MKW_FSR_FG_FLAT_DEPTH=1` supplies a
separate constant depth image to isolate SDK reconstruction; it never changes the game's depth and also does not fix the
shadow defect. These controls are diagnostic experiments, not recommended player settings.
