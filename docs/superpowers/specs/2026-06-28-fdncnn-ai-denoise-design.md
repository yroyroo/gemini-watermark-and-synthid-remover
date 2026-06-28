# FDnCNN AI Denoise — Design Spec

**Date:** 2026-06-28
**Status:** Design approved; pending implementation plan
**Goal:** Add an optional, NCNN/Vulkan-accelerated FDnCNN denoiser as a residual-cleanup method after reverse alpha blending, for still images.

## Context

Reverse alpha blending is mathematically exact on pristine images, but real Gemini outputs are usually resized, recompressed (JPEG), or screenshot-captured *after* watermarking. That post-processing breaks the pixel-perfect alignment between the alpha map and the image, leaving faint sparkle/diamond-edge residuals where the inversion overshoots. The current cleanup (Gaussian / TELEA / Navier-Stokes) uses hand-crafted rules that reduce but can smear these edges.

FDnCNN (Flexible DnCNN, KAIR, MIT) is a learned denoiser (~1.3 MB FP16, ~668K params, 20 conv layers) that reconstructs plausible *clean* content at those edges better than hand-crafted methods. Upstream `allenk/GeminiWatermarkTool` ships it as an optional AI denoise; this spec ports that capability.

## Decisions (locked with user)

- **GPU:** Full Vulkan GPU with automatic CPU fallback.
- **Default behavior:** AI denoise is the **default** cleanup method when built (`WMR_BUILD_AI_DENOISE=ON`); Gaussian is the runtime fallback if GPU/CPU init fails.
- **Build:** Optional `WMR_BUILD_AI_DENOISE` CMake flag, **OFF by default** — the current lean, zero-dep build is unchanged.
- **NCNN source:** NCNN + volk as **git submodules**, built from source with `NCNN_VULKAN=ON`. Self-contained (no vcpkg required for the AI path), version-pinned, matches upstream.
- **Model:** reuse upstream's converted FDnCNN NCNN model (KAIR, MIT), embedded as C byte arrays (~1.3 MB).

## Scope

**IN:** FDnCNN AI denoise as a still-image residual-cleanup method; optional build flag; CLI exposure; guarded tests; docs + license attribution; one optional CI job.

**OUT:** video AI denoise (the video path stays alpha-only — no inpaint); automatic sigma estimation; non-FDnCNN models.

## Architecture / Components

### 1. Build integration — `WMR_BUILD_AI_DENOISE`
- New CMake `option(WMR_BUILD_AI_DENOISE "Build FDnCNN AI denoise (needs Vulkan SDK + NCNN)" OFF)`.
- When ON: `add_subdirectory(external/ncnn)` with `NCNN_VULKAN=ON`, `NCNN_BUILD_EXAMPLES/TOOLS/BENCHMARK=OFF`; link `ncnn`; vendor/link volk; require the Vulkan SDK (provides MoltenVK on macOS, headers/loader elsewhere). Define compile macro `WMR_AI_DENOISE` guarding all AI sources.
- When OFF: no new sources, no new deps — today's build is byte-for-byte unchanged.
- `scripts/build.sh` gains an AI mode (`WMR_AI_DENOISE=1`): init the submodules, verify the Vulkan SDK is present, pass the flag.
- CI: the default release matrix is unchanged; add **one optional AI build job** (macOS) that installs the Vulkan SDK and flips the flag — so ordinary releases stay fast.
- **Verify:** `WMR_BUILD_AI_DENOISE=ON` configures and builds on macOS with the Vulkan SDK; `=OFF` build is identical to today.

### 2. Model embedding
- New isolated TU `src/core/ai_denoise_model.cpp`: the FDnCNN `.param` + `.bin` as `constexpr` byte arrays (produced via ncnn2mem), ~1.3 MB. MIT (KAIR origin).
- README "Third-party" section += KAIR (MIT), NCNN (BSD-3-Clause), volk (MIT).
- **Verify:** the model loads via `ncnn::Net::load_mem(param, param_size, bin, bin_size)` without error.

### 3. NcnnDenoiser module — `src/core/ai_denoise.{hpp,cpp}` (guarded `WMR_AI_DENOISE`)
Port of upstream's `NcnnDenoiser`, PIMPL (`struct Impl`) so ncnn headers stay out of the public header:
- `bool initialize()` — load the embedded model; probe Vulkan via `ncnn::get_gpu_count()`; on ≥1 device use the default GPU, else CPU (`net.opt.num_threads = ncnn::get_cpu_count()`); cache the net. Idempotent.
- `[[nodiscard]] bool is_ready() const`, `bool is_gpu_enabled() const`, `std::string device_name() const`.
- `void denoise(cv::Mat& image, const cv::Rect& region, const cv::Mat& alpha_map, float sigma, float strength, int padding)`:
  1. Build the 4-channel ncnn::Mat input `[R, G, B, σ/255]` (CHW layout; uniform σ map) from the padded ROI.
  2. FDnCNN inference (`net.create_extractor()`; FDnCNN outputs the denoised image directly, not a residual).
  3. Extract the 3-channel RGB result back to a `CV_8UC3` BGR ROI.
  4. Compute the gradient weight mask from `alpha_map` (Sobel magnitude → normalize → sqrt → dilate → blur → scale by `strength`, truncate at 1.0) — **reusing the existing gradient-mask logic** from `inpaint.cpp`.
  5. Per-pixel masked blend over the padded region: `result = mask·denoised + (1−mask)·original`.
- Lazy singleton (one instance; model loads once). Non-copyable, movable.
- **Verify:** `denoise` runs on a synthetic residual; pixels where mask≈0 are unchanged; sparkle-edge pixels are repaired.

### 4. Engine / inpaint wiring
- `enum class InpaintMethod` gains `AiDenoise` (guarded). `inpaint_residual()` dispatches to `NcnnDenoiser::denoise` when `method == AiDenoise`, reusing the shared gradient-mask path.
- Default `InpaintConfig::method = AiDenoise` when `WMR_AI_DENOISE` (compile-time default via `#ifdef`), else `Gaussian`.
- `WatermarkEngine` owns a lazy `NcnnDenoiser` (guarded). At cleanup time: if AI is the selected method **and** the denoiser `is_ready()` → use it; otherwise transparently fall back to Gaussian. A failed Vulkan/init never blocks watermark removal.
- **Verify:** default cleanup uses AI when built; a forced init failure falls back to Gaussian and removal still succeeds.

### 5. CLI — `src/cli/cli_app.{hpp,cpp}`
- `--denoise {ai|soft|ns|telea|off}` (default `ai` when `WMR_AI_DENOISE`, else `soft`); `--sigma <1..150>` (default 50); `--strength <0..300>` (default 120 when AI). Applied to `remove`/`visible` (+ batch forwarding).
- When **not** built with AI: `ai` is absent from `--denoise`, default is `soft`, and `--sigma/--strength` are unused — the CLI reflects the build.
- **Verify:** `wmr remove img --denoise ai` runs; `--denoise off` skips cleanup; a non-AI build rejects `--denoise ai` with a clear message.

### 6. Tests, docs, CI
- `[aidenoise]` ctest tag (compiled only with `WMR_AI_DENOISE`): model loads on CPU; denoise on a synthetic residual; masked blend preserves background; Gaussian fallback path when init fails.
- `README.md` / `CLAUDE.md`: AI-denoise section (build flag, Vulkan-SDK prerequisite, `--denoise/--sigma/--strength`, GPU-vs-CPU behavior). `CHANGELOG.md` entry under `[1.3.0]`.
- Optional CI AI build job.

## Phasing (implementation order)

1. **Build integration** — submodules + CMake flag + Vulkan-SDK docs; get an AI build linking with a model-loading skeleton on macOS.
2. **NcnnDenoiser** — port inference + gradient-mask blend; wire into `inpaint`/`WatermarkEngine` (default-when-built + Gaussian fallback).
3. **CLI + tests + docs + CI job.**

## Risks / Trade-offs

- **Build weight** (Vulkan SDK, NCNN compile, ~1.3 MB model) — mitigated by OFF-by-default + optional CI job.
- **Vulkan availability varies** (CI runners, containers, headless) — CPU fallback guarantees functionality.
- **Over-smoothing at high sigma** — mitigated by edge-only masking + tunable `--sigma`/`--strength`.
- **Video:** no AI denoise (video path is alpha-only); noted as future work.

## Open questions

None blocking. Future work: AI denoise for video frames; automatic sigma estimation from the residual.
