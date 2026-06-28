# Plan: FDnCNN AI Denoise (NCNN/Vulkan) for Still Images

**Status:** Implementation-ready (post 2nd critical review — O1–O4 RESOLVED + folded in; AI CI build chain reconciled to upstream's verified approach)
**Date:** 2026-06-28
**Scope:** Still-image path only. Video AI denoise, automatic sigma estimation, non-FDnCNN models are OUT OF SCOPE.
**Goal:** Add an OPTIONAL FDnCNN AI denoiser (NCNN + Vulkan, CPU fallback) as a still-image residual-cleanup method after reverse alpha blending. OFF-by-default CMake flag (`WMR_BUILD_AI_DENOISE`); when ON, AI is the default cleanup with automatic Gaussian fallback. Ported from upstream `allenk/GeminiWatermarkTool` (MIT). **Ships a separate AI-enabled macOS arm64 release binary (`wmr-macos-arm64-ai`) from a dedicated CI job** alongside the lean, AI-free default binaries (O1 = YES).

---

## 1. Context (Why)

Reverse alpha blending is mathematically exact on pristine inputs, but real Gemini outputs are resized, JPEG-recompressed, or screenshot-captured *after* watermarking, breaking pixel-perfect alignment and leaving faint sparkle/diamond-edge residuals. The current Gaussian/TELEA/NS cleanup uses hand-crafted rules that reduce but can smear these edges.

FDnCNN (Flexible DnCNN, KAIR, MIT) is a learned denoiser (~1.3 MB FP16, ~668K params, 20 conv layers) that reconstructs plausible clean content at those edges better than hand-crafted methods. Upstream ships it as an optional AI denoise; this plan ports that capability behind a flag that leaves today's lean, zero-dep OFF build **provably unchanged**.

This plan is the authoritative design. Where the design spec (`docs/superpowers/specs/2026-06-28-fdncnn-ai-denoise-design.md`) disagreed with verified upstream/local code, **the verified code wins** and the disagreement is recorded in §5.

---

## 2. Scope

### IN
- `WMR_BUILD_AI_DENOISE` CMake option (OFF by default).
- NCNN built from source + volk, gated on the flag. Vulkan SDK required when ON.
- `NcnnDenoiser` module (PIMPL) — ported inference pipeline + masked blend.
- `InpaintMethod::AiDenoise` (guarded); engine default-when-built + Gaussian fallback.
- `--denoise/--sigma/--strength/--radius` CLI on `remove`/`visible` + batch forwarding; graceful non-AI build.
- `src/core/ai_denoise_model.cpp` isolated TU (embedded weights, MIT/KAIR attribution).
- `[aidenoise]` ctest tag (compiled only when flag ON): model load, synthetic residual, masked blend, fallback.
- `scripts/build.sh` AI mode (`WMR_AI_DENOISE=1`).
- One optional CI job (`ai-denoise`) on `macos-14` that **builds AND uploads** a separate release artifact `wmr-macos-arm64-ai` (gated on `v*` tags); added to the release job's `files:` list. The default 4-runner release matrix (lean, AI-free) stays unchanged and AI-free (O1 = YES; see §6.17/§6.18).
- LICENSE/AI-THIRD-PARTY attribution note + README subsection for the AI binary variant (carries KAIR/NCNN/volk licenses).
- Docs (CLAUDE.md, README), CHANGELOG `[1.3.0]`, version bump 1.2.0 → 1.3.0.

### OUT (do NOT touch)
- `src/video/**` — video cleanup stays alpha-only (no inpaint, no AI). Out of scope, future work.
- `src/core/blend_modes.{hpp,cpp}` — reverse-alpha math unchanged.
- `src/synthid/**`, `src/detection/synthid_detector.*`.
- Detection fusion constants / 0.35 threshold.
- The OFF build: no new files compiled, no new deps, no new compile defs, no vcpkg change.

---

## 3. Verified Codebase Facts (authoritative — read during review)

### Local
- **`src/core/inpaint.hpp`** — `enum class InpaintMethod {Gaussian, Telea, NavierStrokes}` (note: `NavierStrokes`, typo-stable — do NOT rename). `struct InpaintConfig {float strength=0.85f; InpaintMethod method=Gaussian; int radius=10; int padding=32; bool full_mask=false;}`. Free function `wmr::inpaint_residual(cv::Mat&, const cv::Rect&, const cv::Mat& alpha_map, const InpaintConfig&)`.
- **`src/core/inpaint.cpp`** Gaussian path (lines 57–109): resize alpha→region; Sobel magnitude; `grad_norm=(grad−min)/(max−min)`; `sqrt`; **`cv::max(grad_weight, alpha_mask)`** where `alpha_mask = threshold(alpha_resized>0.05,1.0)` (interior coverage); dilate **7×7** ellipse; `GaussianBlur(σ=3.0)`; `×strength`; `THRESH_TRUNC@1.0`; embed into padded; `GaussianBlur(σ=1.0)`; blend `weight·blurred + (1−weight)·orig`.
- **`src/core/watermark_engine.{hpp,cpp}`** — `WatermarkEngine()` ctor → `init_alpha_maps()` → `detector_=make_unique<NccDetector>(...)`. Owns alpha maps + `logo_value_=255.0f` + `has_v2_`. `remove_watermark_detected(image, detection, inpaint_strength=0.85f, custom_alpha=nullptr)` → alpha blend → `inpaint_residual(image, detection.region, icfg, custom_alpha)`. `inpaint_residual(...)` member resolves alpha via region-size heuristic then calls the free `wmr::inpaint_residual`. **This is the single cleanup dispatch point AI must hook.**
- **`src/cli/cli_app.hpp`** — `struct CliOptions { ... float inpaint_strength=0.85f; ... bool still_legacy/still_no_legacy; ... }`. No cleanup-method field today.
- **`src/cli/cli_app.cpp`** — `process_single_image` (lines 111–249) calls `engine.remove_watermark_detected(image, detection, opts.inpaint_strength, &alpha)` at line 144. `--inpaint-strength` (Range 0.0–1.0) is the only cleanup knob. APP_VERSION fallback `"1.1.0"` (line 21) — overridden by CMake `APP_VERSION="${PROJECT_VERSION}"`.
- **`src/cli/batch_processor.cpp`** — `process_single` mirrors line 144 at line 71 (`engine.remove_watermark_detected(..., opts.inpaint_strength, &alpha)`).
- **`CMakeLists.txt`** — `project(wmr VERSION 1.2.0 ...)`; `SOURCES` list (lines 9–27); `target_link_libraries` (lines 64–78); macOS/Win platform blocks; `option(WMR_BUILD_TESTS ...)` at line 142.
- **`tests/CMakeLists.txt`** — `TEST_SOURCES` + `LIB_SOURCES` lists; **`noise_residual_subtractor.cpp` is intentionally excluded** from `LIB_SOURCES` (comment at line 25 pattern — unused sources are omitted). `wmr_tests` links Catch2/OpenCV/fmt/spdlog/FFTW3::fftw3f (no FFmpeg → no video/scene_detector).
- **`assets/embedded_assets.hpp`** — pattern: `inline constexpr unsigned char NAME[] = {...}; inline constexpr size_t NAME_size = N;` in `namespace wmr::embedded`. The model TU will follow a sibling pattern in its own namespace.
- **`vcpkg.json`** — version `1.2.0`; deps: opencv4, fmt, cli11, spdlog, fftw3, catch2, ffmpeg[x264]. **No NCNN/volk/vulkan** (AI path is self-contained, NOT via vcpkg — locked decision).
- **`scripts/build.sh`** — macOS/Homebrew, system libs (no vcpkg locally). Verifies `DEPS=(opencv fftw ffmpeg catch2 fmt spdlog cli11)`. CMake configure with `-DWMR_BUILD_TESTS=ON`.
- **`.github/workflows/release.yml`** — verified: `build` job matrix has 4 entries: `macos-14`→`wmr-macos-arm64`, `macos-15`→`wmr-macos-x86_64`, `ubuntu-latest`→`wmr-linux-x86_64`, `windows-latest`→`wmr-windows-x86_64.exe` (the Windows entry sets `vcpkg_triplet: x64-windows-static`). All use `lukka/run-vcpkg@v11` + `-DWMR_BUILD_TESTS=OFF`. The **`release` job** (`needs: build`, `if: startsWith(github.ref,'refs/tags/v')`) **already has `permissions: contents: write`** and uses `softprops/action-gh-release@v2` with `generate_release_notes: true` and a `files:` list of the 4 artifacts. The 4 lean binaries MUST stay AI-free; the AI artifact is a 5th `files:` entry sourced from the new `ai-denoise` job (which the `release` job must `need`).
- **`README.md`** — "Source Projects" (lines 215–221) credits allenk repos; License MIT (line 225). No NCNN/KAIR/volk attribution yet.
- **Version** — CMake + vcpkg.json at `1.2.0`; tags `v1.0.0`/`v1.1.0`/`v1.2.0`. This plan bumps to **1.3.0**.

### Upstream (`allenk/GeminiWatermarkTool` master, verified via raw GitHub)
- **`src/core/ai_denoise.hpp`** — `class NcnnDenoiser` under `#ifdef GWT_HAS_AI_DENOISE`. PIMPL (`struct Impl`). Non-copyable, movable. `bool initialize()`, `[[nodiscard]] bool is_ready() const`, `bool is_gpu_enabled() const`, `std::string device_name() const`, `denoise(image, region, alpha_map, sigma=25.0f, strength=0.85f, padding=16)`. **Note: upstream default sigma=25, strength=0.85 in the signature, but the CLI overrides with 50/1.2.**
- **`src/core/ai_denoise.cpp`** — the core port. Key verified facts:
  - **Blob indices** are hardcoded: `BLOB_INPUT=0`, `BLOB_OUTPUT=20` (binary param, no string names). Extractor uses `ex.input(0,...)` / `ex.extract(20,...)`.
  - **Model load API**: `net.load_param(const unsigned char*)` and `net.load_model(const unsigned char*)` — each returns **bytes consumed (>0 = success)**. **NOT** `load_mem` (the spec was wrong here).
  - **GPU init**: `vulkan_loader_present()` probes via `dlopen`/`LoadLibraryA` FIRST (prevents the documented SIGBUS crash in NCNN issues #30/#31 when the loader is absent); only then `ncnn::create_gpu_instance()`; `ncnn::get_gpu_count()`, `ncnn::get_default_gpu_index()`, `ncnn::get_gpu_info(idx).device_name()`; `net.opt.use_vulkan_compute=true`. CPU fallback: `net.opt.num_threads=ncnn::get_cpu_count()`.
  - **FP16**: `use_fp16_packed=true`, `use_fp16_storage=true`, `use_fp16_arithmetic=false`.
  - **`build_input`**: `ncnn::Mat(w,h,4)` (CHW), channels `[R,G,B,σ/255]` uniform; input is RGB float (BGR→RGB convert, /255).
  - **`run_inference`**: FDnCNN outputs the clean image directly (NOT a residual); clamp [0,1]; back to BGR uint8.
  - **`compute_gradient_mask`** — **NOT identical to our `inpaint.cpp` Gaussian mask**: resize; Sobel; normalize; `sqrt`; dilate **5×5** ellipse; `GaussianBlur(σ=2.0)`; `×strength`; `THRESH_TRUNC@1.0`. **Omits our `cv::max(grad_weight, alpha_mask)` interior-coverage step.** Returns mask sized to `(region_w, region_h)` (not padded).
  - **`denoise`**: clamp sigma 0–150, strength 0–3.0; compute mask; build padded ROI (±padding, clamped); `run_inference` on padded ROI; embed mask into padded coords; `GaussianBlur(σ=1.0)`; blend `weight·denoised+(1−weight)·orig`; write back.
  - **Padding default 16** in signature; the CLI calls with **32**.
- **`src/core/ncnn_shim.hpp`** — must be included FIRST before any ncnn header: `#include <volk.h>`; `#define NCNN_SIMPLEVK_H` (skip ncnn's internal `simplevk.h` to avoid duplicate Vulkan symbol defs when statically linking); then `#include "net.h"/"gpu.h"/"cpu.h"`. **Decision: port this shim verbatim** (volk from vcpkg's `ai-denoise` feature provides `<volk.h>`; R8).
- **`src/core/ai_denoise_model.cpp`** — isolated TU. `#include "model_core.id.h"` + `#include "model_core.mem.h"` (generated by `ncnn2mem`). Exposes `gwt::ai_model::param_data()` / `bin_data()` returning `const unsigned char*` to the two `S__...` symbols. **CRITICAL: the `.mem.h`/`.id.h` headers are GENERATED artifacts, gitignored upstream, NOT committed to the repo tree.** See §7 provenance.
- **`CMakeLists.txt`** (upstream) — `option(ENABLE_AI_DENOISE ...)`. When ON: NCNN source at `external/ncnn/ncnn-20260113-src` (a **git submodule** at that path, checked out at the NCNN release tag — NOT a separate downloaded tarball; verified in upstream `.gitmodules`); `add_subdirectory(${NCNN_ROOT} ncnn)`; `NCNN_VULKAN ON`, `NCNN_VULKAN_ONLINE_SPIRV OFF`, `NCNN_SHARED_LIB OFF`, `NCNN_OPENMP OFF`, `NCNN_BUILD_{TESTS,TOOLS,EXAMPLES,BENCHMARK} OFF` (`NCNN_BENCHMARK` set too), `NCNN_INSTALL_SDK OFF`, `NCNN_ENABLE_LTO ON`, minimal `WITH_LAYER_*` set (convolution, relu, input, concat, split, packing, cast, flatten, padding, binaryop); `find_package(volk CONFIG REQUIRED)` **(volk via VCPKG, via the `ai-denoise` manifest feature — NOT a submodule, NOT vendored)**; compile def `GWT_HAS_AI_DENOISE=1`. MSVC: `NCNN_DISABLE_RTTI/EXCEPTION OFF`. **NOTE: upstream's volk + vcpkg vulkan-headers supply the Vulkan types/loader at build time; NCNN ships its own MoltenVK-capable loader at runtime via its internal simplevk. No `brew install --cask vulkan-sdk` is used in upstream CI.**
- **`vcpkg.json`** (upstream) — `ai-denoise` feature depends on `volk` only. NCNN itself is built from the submodule, not vcpkg. **This is the mechanism O1's AI CI job mirrors** (a vcpkg manifest feature is opt-in, so the lean default binaries built WITHOUT the feature stay AI-free).
- **`src/core/watermark_engine.hpp`** (upstream) — `enum class InpaintMethod {GAUSSIAN, TELEA, NS, AI_DENOISE}` — AI_DENOISE is the LAST value, `#ifdef GWT_HAS_AI_DENOISE`-guarded in the enum itself.
- **`src/cli/cli_app.cpp`** (upstream) — flags: `--denoise` choices `ai,ns,telea,soft,off` (default `off` upstream); `--sigma` Range 1–150 default 50; `--strength` Range 0–300 **percent** default 120 (AI) / 85 (others) → divided by 100 internally; `--radius` Range 1–25 default 10. `DenoiseConfig::from_string` resolves method; AI init failure → fallback to NS at strength 0.85.
- **`.github/workflows/build.yml`** (upstream — verified, this is the model O1 mirrors) — upstream ships AI-enabled binaries via a **separate matrix/job variant** (`gwt-mini-*`) alongside leaner ones, all in one workflow. The AI build chain is: `actions/checkout@v4` with `submodules: recursive`; `lukka/run-vcpkg@v11`; configure with `-DENABLE_AI_DENOISE=ON -DVCPKG_MANIFEST_FEATURES="ai-denoise" -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake`; build. **No Vulkan SDK install step on any platform** (volk + vcpkg's vulkan headers + NCNN's internal runtime loader suffice; MoltenVK is bundled with the macOS runtime). The `release` job downloads all artifacts, zips each, generates `SHA256SUMS.txt`, and uploads via `softprops/action-gh-release@v1` with a `files:` list. **This is the proven working approach — O1's AI CI job must mirror it** (submodules for NCNN + vcpkg manifest feature for volk + NO Vulkan SDK install).

---

## 4. Critical Review Findings & Fixes (by criterion)

### 4.1 Correctness

**C1 — Model-load API (CRITICAL correction of the spec).** The spec said `ncnn::Net::load_mem(param, param_size, bin, bin_size)`. **Wrong.** Verified upstream uses `int ret_param = net.load_param(param_ptr); int ret_model = net.load_model(bin_ptr);` where each returns bytes consumed (>0 = success, ≤0 = failure). Fix: the plan uses `load_param`/`load_model` with the `>0` success check, exactly as upstream.

**C2 — Gradient-mask reuse-vs-port (CRITICAL resolution).** The spec said "reuse the existing gradient-mask logic from `inpaint.cpp`." Verified: upstream `compute_gradient_mask` and our `inpaint.cpp` Gaussian mask **diverge** in three ways:
  - Ours adds `cv::max(grad_weight, alpha_mask)` to cover the watermark *interior* (alpha>0.05); upstream does not (edge-only).
  - Dilation: ours 7×7, upstream 5×5.
  - Blur σ: ours 3.0, upstream 2.0.
  - Ours embeds into a *padded* coordinate system then re-blurs; upstream returns a region-sized mask and embeds in `denoise`.
  **Decision: PORT upstream's `compute_gradient_mask` verbatim into `NcnnDenoiser`** (do NOT reuse `inpaint.cpp`). Rationale: (a) AI denoise intentionally repairs *edges only* — adding interior coverage would over-smooth the whole watermark region, defeating the edge-only design; (b) the tuned constants (5×5/σ2.0) are co-calibrated with FDnCNN's output; reusing ours would silently change behavior; (c) keeping the AI mask self-contained means the AI TU has zero coupling to `inpaint.cpp` internals (which are not part of any stable API). The duplicated Sobel/normalize/sqrt/dilate/blur is ~15 lines — acceptable, explicitly justified, and matches upstream exactly (faithful port). Document the divergence in a code comment so a future refactor doesn't naively DRY them.

**C3 — 4-channel CHW input + sigma normalization.** Verified: `ncnn::Mat(w,h,4)`, channel strides `input.channel(k)` are contiguous H·W blocks, channels `[R,G,B,σ/255]` (uniform σ map), input pre-converted BGR→RGB then /255. Output `[R,G,B]`→BGR uint8. The plan ports `build_input`/`extract_output`/`run_inference` verbatim. Blob indices `0` (in) and `20` (out) are hardcoded.

**C4 — Masked blend formula.** Verified `result = weight·denoised + (1−weight)·original` (per-channel via 3-ch merge), weight embedded in padded coords + `GaussianBlur(σ=1.0)`. Ported verbatim.

**C5 — Default-when-built + fallback.** When `WMR_AI_DENOISE`: default `InpaintConfig::method=AiDenoise` (compile-time `#ifdef`). At cleanup time the engine lazily inits `NcnnDenoiser`; if `is_ready()` is false (init failed / no Vulkan+CPU), **transparently fall back to Gaussian** — removal always succeeds. This matches upstream's `denoiser->is_ready()` gate.

**C6 — GPU probe ordering.** Verified: `vulkan_loader_present()` (dlopen/LoadLibrary probe) MUST run before `ncnn::create_gpu_instance()` to avoid the documented SIGBUS on loader-less systems (NCNN issues #30/#31). The plan ports `vulkan_loader_present()` verbatim (the macOS MoltenVK note included). On probe failure → CPU path, no crash.

### 4.2 Regressions (every guard enumerated)

**R1 — Default-when-built is a behavior change.** Anyone who flips `WMR_BUILD_AI_DENOISE=ON` gets AI as default cleanup instead of Gaussian. **Mitigation:** (a) OFF-by-default, so the shipped/release build (flag OFF) is byte-for-byte unchanged; (b) the AI path falls back to Gaussian on any init failure; (c) documented in CLAUDE.md/README; (d) `--denoise soft` / `--denoise off` let users force the old behavior. Net: only opt-in builders are affected, and they always have a working path.

**R2 — OFF build provably unchanged (the hardest requirement).** Every AI symbol/reference must be guarded so the OFF build compiles with zero AI knowledge. Concrete guards:
  - `InpaintMethod::AiDenoise` enum value wrapped in `#ifdef WMR_AI_DENOISE` inside the enum (matches upstream's `GWT_HAS_AI_DENOISE` pattern). The `switch`/`if` on `method` in `inpaint.cpp`/engine has an `#ifdef`-guarded `case AiDenoise:` that is absent when OFF.
  - `WatermarkEngine`'s `std::unique_ptr<NcnnDenoiser> denoiser_` field + its lazy-init call site are `#ifdef WMR_AI_DENOISE`-guarded.
  - `InpaintConfig::method` default is `#ifdef WMR_AI_DENOISE ? AiDenoise : Gaussian`.
  - `#include "core/ai_denoise.hpp"` in `watermark_engine.cpp`/`cli_app.cpp` guarded.
  - `--denoise/--sigma/--strength/--radius` CLI options registered only under `#ifdef WMR_AI_DENOISE`; the `CliOptions` fields they bind to are always present (cheap) but unused when OFF, so no dead-symbol warnings.
  - **No new `#include` of ncnn/volk headers leaks into any non-guarded TU.** ncnn headers live only behind `ai_denoise.cpp` + `ai_denoise_model.cpp` + `ncnn_shim.hpp`, all conditionally compiled.
  - Acceptance: `cmake -B build -S . -DWMR_BUILD_AI_DENOISE=OFF` (the default) produces a binary whose sources/deps/defs are identical to today. Verified by diffing the cmake configure summary + `ninja -t commands`.

**R3 — Model TU conditionally added to BOTH CMakeLists.** `src/core/ai_denoise_model.cpp` (~1.3 MB+ of arrays) and `src/core/ai_denoise.cpp` are added to `CMakeLists.txt` `SOURCES` and `tests/CMakeLists.txt` `LIB_SOURCES` **only inside `if(WMR_BUILD_AI_DENOISE)`**. When OFF: absent from both, not compiled, no ncnn link. This satisfies CLAUDE.md's "new TUs must be added to BOTH" rule while keeping the OFF build clean.

**R4 — Vulkan/init failure never blocks removal.** `NcnnDenoiser::initialize()` returns false on any failure (loader absent, model corrupt, OOM); the engine checks `is_ready()` and routes to Gaussian. `denoise()` itself no-ops + logs if `!ready` (defensive). A thrown exception in init is caught at the engine boundary and logged → Gaussian. No `throw` propagates to `process_single_image`.

**R5 — Lazy singleton thread-safety.** Not required for correctness today (single-threaded CLI/batch), but the lazy `NcnnDenoiser` is constructed via `std::call_once` / function-local static so a future batch-parallel caller is safe. NCNN `ncnn::Net` is itself not thread-safe across instances sharing a GPU context, so we keep ONE process-wide instance. Documented "one instance, not thread-safe for concurrent `denoise` calls" (matches upstream header comment).

**R6 — ncnn_shim port.** `NCNN_SIMPLEVK_H` define + `#include <volk.h>` ordering is mandatory to avoid duplicate Vulkan loader symbols when statically linking NCNN. Port `ncnn_shim.hpp` verbatim (renamed guards to `WMR_AI_DENOISE`). Verified necessary, not optional.

**R7 — CI isolation: the 4 default release binaries MUST stay AI-free (O1 regression guard).** The new `ai-denoise` job is a SEPARATE job (not a matrix entry of `build`). The `build` matrix's 4 jobs continue to configure with NO `-DWMR_BUILD_AI_DENOISE`, NO submodules checkout (their `actions/checkout` stays without `submodules:`), and NO `VCPKG_MANIFEST_FEATURES`. volk is added to `vcpkg.json` ONLY behind the new `ai-denoise` manifest feature (a vcpkg feature is opt-in — `run-vcpkg` with default features will NOT pull volk). Concrete guards:
  - `build` matrix jobs: checkout WITHOUT `submodules: recursive` (no NCNN fetched → even a stray flag reference fails fast); configure line unchanged (no AI flag); `VCPKG_MANIFEST_FEATURES` unset (default features = the existing 7 deps only).
  - `ai-denoise` job: checkout WITH `submodules: recursive`; `VCPKG_MANIFEST_FEATURES: "ai-denoise"` (pulls volk); `-DWMR_BUILD_AI_DENOISE=ON`; uploads ONLY its own artifact.
  - `release` job: `needs: [build, ai-denoise]`; `files:` list = the 4 lean artifacts + the 1 AI artifact. The AI artifact's distinct name (`wmr-macos-arm64-ai`) means there is no collision/overwrite with `wmr-macos-arm64`.
  - Acceptance: a tag push produces 5 artifacts; the 4 lean ones are byte-identical to a pre-AI release (verified by checking `nm`/`otool -L` for absence of ncnn/volk/MoltenVK symbols in the lean binaries — see §9).

**R8 — volk provenance divergence from the first-pass plan (CORRECTION).** The first-pass plan said "volk via git submodule." Verified upstream + upstream CI use **volk via vcpkg** (`find_package(volk CONFIG REQUIRED)` + the `ai-denoise` manifest feature). This is corrected throughout the plan: volk is a vcpkg manifest-feature dependency, NOT a submodule. NCNN remains a git submodule (matches upstream `.gitmodules`). This keeps volk versioning inside vcpkg's baseline (reproducible) and matches the proven working CI build.

### 4.3 Best Practices
- **PIMPL**: `NcnnDenoiser::Impl` holds `ncnn::Net`; the public `ai_denoise.hpp` includes only `<opencv2/core.hpp>` + `<memory>` + `<string>`. ncnn/volk headers never escape `ai_denoise.cpp`/`ai_denoise_model.cpp`. This keeps compile times and the include graph clean.
- **Idiomatic C++20**: `= delete` copy, `= default` move, `[[nodiscard]]`, `std::unique_ptr` RAII for `ncnn::Net` (destructor releases GPU instance), `constexpr` blob indices.
- **RAII for `ncnn::Net`**: member of `Impl`; destroyed with `Impl`. GPU instance released by ncnn's destructor.
- **CMake target-level guards**: `target_link_libraries(${PROJECT_NAME} PRIVATE ncnn volk::volk)` and `find_package(volk ...)` + `add_subdirectory(external/ncnn/...)` only inside `if(WMR_BUILD_AI_DENOISE)`. `target_compile_definitions(... WMR_AI_DENOISE=1)` likewise guarded via a generator expression so OFF never sees it.
- **License attribution (now mandatory for the SHIPPED binary — O1)**: Because the AI binary redistributes NCNN (BSD-3-Clause), volk (MIT), and the FDnCNN model (KAIR, MIT) compiled in, the attribution MUST travel with the binary, not just the repo:
  - README "Source Projects" += KAIR (FDnCNN, MIT), Tencent/ncnn (BSD-3-Clause), zeux/volk (MIT).
  - A new top-level `LICENSE-AI.md` (or `assets/AI-THIRD-PARTY.md`) committed to the repo, reproducing the NCNN BSD-3-Clause notice + volk MIT notice + KAIR/FDnCNN MIT notice + the allenk/GeminiWatermarkTool conversion credit. **This file is added to the AI artifact's release zip** (the `ai-denoise` CI job copies it next to the binary before upload) so users downloading `wmr-macos-arm64-ai` receive the licenses.
  - A copy/license note header in `ai_denoise_model.cpp` and `ai_denoise.cpp`.
  - The release notes / CHANGELOG `[1.3.0]` entry links to `LICENSE-AI.md` and names the AI artifact explicitly so it is discoverable.
- **No scope creep**: no `guided_detect`, no `--region/--snap/--fallback-region` (upstream CLI has them; our CLI doesn't and they're unrelated to AI). `--denoise/--sigma/--strength/--radius` only.

### 4.4 Completeness

**M1 — Submodule + vcpkg-feature setup (corrected to match upstream's verified working chain).** NCNN is a git submodule (matches upstream `.gitmodules` → `external/ncnn/ncnn-20260113-src`). volk comes from **vcpkg** via a new `ai-denoise` manifest feature (NOT a submodule — R8). Concretely:
  ```bash
  # NCNN submodule — pin to the real Tencent/ncnn release tag matching upstream's 20260113
  git submodule add https://github.com/Tencent/ncnn.git external/ncnn/ncnn-20260113-src
  git -C external/ncnn/ncnn-20260113-src checkout 20260113   # NCNN release tag
  ```
  volk is declared in `vcpkg.json` under a new feature:
  ```json
  "features": {
    "ai-denoise": {
      "description": "FDnCNN AI denoise (NCNN + Vulkan); requires WMR_BUILD_AI_DENOISE=ON + the NCNN submodule",
      "dependencies": [ "volk" ]
    }
  }
  ```
  > NOTE for the implementer: pin the exact NCNN tag/SHA at execution time; verify `external/ncnn/ncnn-20260113-src/CMakeLists.txt` exists and exposes `NCNN_VULKAN`/`WITH_LAYER_*` cache options before building. If the `20260113` tag doesn't exist on the public ncnn repo, use the nearest real release tag (e.g. `20240820`) and record the actual SHA in `.gitmodules` + the CHANGELOG. The exact tag is a build-detail, not a design decision (O2). **The submodule path `external/ncnn/ncnn-20260113-src` is intentional** — it matches upstream verbatim, so the NCNN CMake block can be a faithful port. **The lean default binaries never fetch the submodule** (their CI checkout omits `submodules:`), and `ai-denoise` is a vcpkg feature that defaults OFF, so volk is never pulled for them (R7).

**M2 — Vulkan build-time detection.** NCNN's `NCNN_VULKAN=ON` + volk need Vulkan *headers* at build time (supplied by vcpkg's volk/vulkan-headers transitively, as in upstream CI) and a Vulkan *loader* at runtime (MoltenVK on macOS, shipped by the OS). **Upstream CI does NOT `brew install --cask vulkan-sdk` and we mirror that** (verified). Therefore the CMake guard is `find_package(Vulkan QUIET)` only as a *diagnostic* (clear error if truly absent), but the expected path is that vcpkg's volk brings the headers. For local non-vcpkg builds (the `scripts/build.sh` system-Homebrew path), volk + the runtime loader come from Homebrew: `brew install vulkan-volk moltenvk` — **`vulkan-volk`** (zeux/volk) ships `volkConfig.cmake` + the Vulkan headers so `find_package(volk CONFIG)` resolves; **`moltenvk`** provides the runtime `libvulkan`/MoltenVK loader. (Do NOT use Homebrew's `volk` formula — that is GNU Radio's vector lib — nor the `vulkan-sdk` cask, which lacks volk's cmake config.) The `build.sh` AI mode (M5) adds `vulkan-volk` to the dep + `CMAKE_PREFIX_PATH` list.

**M3 — Model provenance (RESOLVED — direct fetch, no upstream build).** The `.mem.h`/`.id.h` headers ARE **committed in GWT** at `external/ncnn/model-convert/output/` (VERIFIED — contrary to the pass-1/2 assumption that they were gitignored). **DONE: fetched directly** into our `assets/model_core.{mem.h,id.h}`. `model_core.mem.h` is 6.5 MB (arrays `S__CodeForge_tools_gemini_watermark_tool_release_repo_external_ncnn_model_convert_output_model_core_fp16_param_bin` + `..._fp16_bin`); `model_core.id.h` (1.8 KB) defines `BLOB_in0=0` / `BLOB_out0=20` (confirms the hardcoded blob indices in §6.3). The original "build upstream once" options are moot. **Resolution:** obtain the two generated headers as **build artifacts** by ONE of these, in priority order:
  1. **Preferred:** build upstream once locally (or in the optional CI job) with `ENABLE_AI_DENOISE=ON`, which runs `ncnn2mem` and emits `external/ncnn/model-convert/output/model_core.{mem.h,id.h}`; **copy those two generated files into our `assets/`** (committed, MIT/KAIR-attributed). This is the faithful path — the bytes are upstream's, just captured post-generation.
  2. **Fallback:** fetch them from an upstream **release asset** or a CI artifact if upstream publishes one.
  3. **Last resort:** document a one-time `ncnn2mem` invocation against the FDnCNN `.param`/`.bin` (KAIR-provided) for whoever sets up the AI build, and pin the resulting bytes.
  Whichever path, the committed `ai_denoise_model.cpp` includes `assets/model_core.mem.h` + `assets/model_core.id.h` and exposes `wmr::ai_model::param_data()`/`bin_data()` to the (possibly path-mangled) `S__...` symbols. **Attribution header: "FDnCNN model converted to NCNN by allenk/GeminiWatermarkTool (MIT); original model from cszn/KAIR (MIT)."** The implementer MUST record which provenance path was used in the CHANGELOG and commit message.
  **O3 DECIDED: commit both headers definitively (~6.8 MB total).** No "revisit if clone times suffer" hedge. The repo-size impact is noted in the CHANGELOG `[1.3.0]` (e.g. "Adds ~6.8 MB of embedded-model headers; required for `WMR_BUILD_AI_DENOISE=ON`."). Self-contained, reproducible, matches the "embedded" intent — and now required because the AI CI job (O1) must build from a clean checkout without a network step to fetch the model.

**M4 — ncnn_shim decision: PORT.** volk comes from vcpkg (provides `<volk.h>`); port `ncnn_shim.hpp` verbatim (guard renamed `GWT_HAS_AI_DENOISE`→`WMR_AI_DENOISE`). Not optional (R6).

**M5 — `scripts/build.sh` AI mode.** Add an `WMR_AI_DENOISE=1` branch: `git submodule update --init --recursive`; ensure `brew install vulkan-volk moltenvk` (`vulkan-volk` = zeux/volk → `volkConfig.cmake` + Vulkan headers; `moltenvk` = runtime `libvulkan`/MoltenVK; NOT the GNU-Radio `volk`, NOT the `vulkan-sdk` cask); add `vulkan-volk` to the dep list + `CMAKE_PREFIX_PATH` so `find_package(volk CONFIG)` resolves; append `-DWMR_BUILD_AI_DENOISE=ON` to the cmake configure. When unset/0: unchanged (flag defaults OFF).

**M6 — Dedicated `ai-denoise` CI job that BUILDS AND UPLOADS (O1 = YES).** New `ai-denoise` job in `release.yml`, sibling of `build`. Platform scope: **macOS arm64 only** (`runs-on: macos-14`) for the first AI release — x86_64/Linux/Windows AI builds are explicitly future work (the FDnCNN+NCNN+Vulkan matrix is heavier and macOS arm64 covers the primary dev/user platform first). The job mirrors upstream's verified `build.yml` AI chain (submodules + vcpkg `ai-denoise` feature, NO Vulkan SDK install). Full job shape (§6.17 has the exact YAML):
  - `needs:` none (parallel with `build`); `runs-on: macos-14`.
  - `actions/checkout@v4` **with `submodules: recursive`** (fetches the NCNN submodule).
  - `lukka/run-vcpkg@v11` with `vcpkgJsonGlob: '**/vcpkg.json'` and `runVcpkgInstall: true`. **Critical:** the configure step passes `-DVCPKG_MANIFEST_FEATURES="ai-denoise"` so vcpkg pulls volk (and its transitive vulkan headers) — without this flag volk is absent and `find_package(volk)` fails. The lean `build` jobs do NOT pass this flag, so they stay AI-free (R7).
  - Configure: `cmake -B build -S . -GNinja -DCMAKE_BUILD_TYPE=Release -DWMR_BUILD_TESTS=OFF -DWMR_BUILD_AI_DENOISE=ON -DVCPKG_MANIFEST_FEATURES="ai-denoise" -DCMAKE_TOOLCHAIN_FILE=${{ github.workspace }}/vcpkg/scripts/buildsystems/vcpkg.cmake`. (Tests OFF for the shipped artifact, matching the lean jobs; the `[aidenoise]` tests run in the developer ON build / a separate non-release workflow_dispatch run.)
  - Build: `cmake --build build --config Release`.
  - **Smoke test (recommended, optional):** `./build/wmr --version` (confirms the binary launches; AI init is lazy so a full denoise smoke test needs test images — keep it to `--version` in CI to avoid coupling the release to test-data presence, matching the lean jobs which also run no tests).
  - Package: `mkdir -p dist && cp build/wmr dist/wmr-macos-arm64-ai && chmod +x dist/wmr-macos-arm64-ai` AND copy `LICENSE-AI.md` (§4.3) next to it so the licenses travel with the binary.
  - `actions/upload-artifact@v4` with `name: wmr-macos-arm64-ai`, `path: dist/` (the binary + the license note).
  - Gating: the job runs on every push to be safe for testing, but its artifact only reaches a release via the `release` job (which is `if: startsWith(github.ref,'refs/tags/v')`). To avoid burning CI on every PR for a heavy job, gate with `if: startsWith(github.ref, 'refs/tags/v') || github.event_name == 'workflow_dispatch'` so it runs on tags + manual dispatch only.
  - The 4-job `build` matrix is **untouched** (no `WMR_BUILD_AI_DENOISE`, no submodules, no `ai-denoise` feature).

**M7 — Conditional source lists.** In `CMakeLists.txt`: `if(WMR_BUILD_AI_DENOISE) list(APPEND SOURCES src/core/ai_denoise.cpp src/core/ai_denoise_model.cpp) ... endif()`. Mirror in `tests/CMakeLists.txt` `LIB_SOURCES`.

**M8 — CLI defaults/ranges (match upstream).** `--sigma` Range 1–150 default 50; `--strength` Range 0–300 **percent** default 120 (AI) / 85 (others), internally `/100` → fraction; `--radius` Range 1–25 default 10; `--denoise` choices `ai|ns|telea|soft|off`, default `ai` when `WMR_AI_DENOISE` else `soft`. Validation: invalid `--denoise` value → CLI11 error; `--sigma`/`--strength` out of range → CLI11 Range error; `ai` selected on a non-AI build → clear error message + exit 2.

**M9 — Tests.** New `tests/unit/ai_denoise_test.cpp` under `#ifdef WMR_AI_DENOISE` (file-level guard so the OFF build compiles an empty TU — or omit the file from `TEST_SOURCES` when OFF). Tag `[aidenoise]`: (1) model loads on CPU (`initialize()` true, `is_gpu_enabled()` true/false, `is_ready()` true); (2) denoise on a synthetic residual (forward-blend alpha → reverse-blend → AI denoise → assert sparkle pixels repaired, background ≤ a tolerance unchanged); (3) masked-blend preserves background (mask≈0 region byte-identical pre/post); (4) fallback path (force `initialize()` failure → engine routes to Gaussian, removal still succeeds).

**M10 — Version bump + docs.** CMake + vcpkg.json `1.2.0`→`1.3.0`. CLAUDE.md AI section. README AI section + attribution. CHANGELOG `[1.3.0]`.

### 4.5 Integration
- **DRY:** the AI mask is intentionally NOT shared with `inpaint.cpp` (C2). The Sobel/normalize/sqrt is ~15 lines duplicated with a documented justification comment — not a violation of DRY because the two masks have deliberately different constants and coverage semantics.
- **`InpaintMethod::AiDenoise`** guarded in-enum (upstream pattern).
- **`WatermarkEngine`** lazy `NcnnDenoiser` (guarded field); cleanup dispatch: `#ifdef WMR_AI_DENOISE if (cfg.method==AiDenoise && denoiser_->is_ready()) { denoiser_->denoise(...); return; } #endif` before the Gaussian/Telea/NS switch.
- **CLI wiring** + graceful non-AI build; batch forwards the same fields.
- **Conditional sources in BOTH CMakeLists.**
- **Dedicated `ai-denoise` CI job builds + uploads a separate AI artifact** (`wmr-macos-arm64-ai`); the 4 default `build` matrix jobs stay AI-free and untouched. The `release` job `needs: [build, ai-denoise]` and adds the AI artifact to its `files:` list (O1). The AI artifact's distinct name guarantees no collision with `wmr-macos-arm64` (R7).

---

## 5. Spec vs. Verified-Code Divergences (corrections applied)

| # | Spec claim | Verified truth | Plan's resolution |
|---|---|---|---|
| D1 | Model loads via `ncnn::Net::load_mem(param,param_size,bin,bin_size)` | Upstream uses `load_param(ptr)` + `load_model(ptr)`, each returning bytes consumed (>0=ok) | Use `load_param`/`load_model` with `>0` check (C1) |
| D2 | "Reuse the existing gradient-mask logic from `inpaint.cpp`" | Upstream `compute_gradient_mask` differs (no interior `alpha_mask` coverage; 5×5 dilate; σ2.0 blur) and returns region-sized, not padded | PORT upstream's mask verbatim into `NcnnDenoiser`; do NOT reuse (C2) |
| D3 | "NCNN + volk as git submodules" (first-pass locked decision) | Verified: upstream uses an NCNN **git submodule** at `external/ncnn/ncnn-20260113-src` + volk from **vcpkg** (via the `ai-denoise` manifest feature). NOT a separate tarball, and volk is NOT a submodule. | NCNN as a git submodule at the upstream-matching path; volk via the vcpkg `ai-denoise` feature (M1, R8). The first-pass plan's "volk submodule" was wrong and is corrected. |
| D4 | Model embedded as "C byte arrays (produced via ncnn2mem)" implying they're in the repo | `.mem.h`/`.id.h` are GENERATED, gitignored, NOT in the upstream tree | Capture generated headers from a one-time upstream build into `assets/` (M3) |
| D5 | (implied) `--strength` default "120% for AI" as a fraction | Upstream `--strength` is **percent 0–300**, `/100` internally; AI default 120 → 1.2 fraction | CLI stores percent, divides by 100 (M8) |
| D6 | `NcnnDenoiser::denoise` padding "32" | Upstream signature default 16, CLI calls with 32 | Keep signature default 16, CLI/engine pass 32 (matches upstream CLI) |
| D7 | Enum value name not specified | Upstream uses `AI_DENOISE` (SHOUTY); our codebase is `PascalCase` | Use `AiDenoise` (codebase convention; upstream's shouty style is GUI-app residue) |

---

## 6. File-by-File Changes

### 6.1 New: `src/core/ncnn_shim.hpp`
Verbatim port of upstream `ncnn_shim.hpp`, guards renamed `GWT_HAS_AI_DENOISE`→`WMR_AI_DENOISE`. Contents: platform `VK_USE_PLATFORM_*` guard (Android); `#include <volk.h>`; `#define NCNN_SIMPLEVK_H`; `#include "net.h"`,`"gpu.h"`,`"cpu.h"`. Entire file wrapped in `#ifdef WMR_AI_DENOISE`.

### 6.2 New: `src/core/ai_denoise.hpp`
Guarded `#ifdef WMR_AI_DENOISE`. Namespace `wmr`. `class NcnnDenoiser` with PIMPL (`struct Impl`), non-copyable, movable. Methods: `bool initialize()`, `[[nodiscard]] bool is_ready() const`, `bool is_gpu_enabled() const`, `std::string device_name() const`, `void denoise(cv::Mat& image, const cv::Rect& region, const cv::Mat& alpha_map, float sigma=25.0f, float strength=0.85f, int padding=16)`. Includes only `<opencv2/core.hpp>`,`<memory>`,`<string>`. (Signature defaults match upstream; CLI overrides to 50/1.2/32.)

### 6.3 New: `src/core/ai_denoise.cpp`
Guarded `#ifdef WMR_AI_DENOISE`. Port upstream `ai_denoise.cpp` adapting namespace `gwt`→`wmr`, includes `"core/ncnn_shim.hpp"` FIRST then `"core/ai_denoise.hpp"`, `gwt::ai_model`→`wmr::ai_model`. Verbatim ports of: `vulkan_loader_present()`, blob-index constants (`BLOB_INPUT=0`,`BLOB_OUTPUT=20`), `Impl` (`load_model` with `load_param`/`load_model` `>0` checks; `init_gpu`/`init_cpu`; `build_input`/`extract_output`/`run_inference`; `compute_gradient_mask`), public API. Attribution header: MIT, credits allenk/GeminiWatermarkTool + KAIR.

### 6.4 New: `src/core/ai_denoise_model.cpp`
Guarded `#ifdef WMR_AI_DENOISE`. Includes `"model_core.id.h"` + `"model_core.mem.h"` (both committed under `assets/`, found via the `assets/` include dir already on the target). Exposes:
```cpp
namespace wmr::ai_model {
const unsigned char* param_data();  // -> S__..._param_bin
const unsigned char* bin_data();    // -> S__..._bin
}
```
The `S__...` symbol names come from the captured `model_core.id.h` (path-mangled by ncnn2mem). Attribution header: FDnCNN/KAIR (MIT) + allenk conversion (MIT). **Provenance path recorded in CHANGELOG (M3).**

### 6.5 New: `assets/model_core.mem.h` + `assets/model_core.id.h`
The two generated headers, captured per M3. ~6.8 MB text / ~1.3 MB weights. Committed. (If repo size is a concern, the implementer may `git commit` them and note the size in the CHANGELOG; they are required for the ON build.)

### 6.6 Edit: `src/core/inpaint.hpp`
Add the guarded enum value (upstream pattern, last position):
```cpp
enum class InpaintMethod {
    Gaussian,
    Telea,
    NavierStrokes
#ifdef WMR_AI_DENOISE
    , AiDenoise   // FDnCNN NCNN/Vulkan AI denoise (edge-only masked blend)
#endif
};
```
Extend `InpaintConfig` default:
```cpp
struct InpaintConfig {
    float strength = 0.85f;
    InpaintMethod method =
#ifdef WMR_AI_DENOISE
        InpaintMethod::AiDenoise;
#else
        InpaintMethod::Gaussian;
#endif
    int radius = 10;
    int padding = 32;
    bool full_mask = false;
};
```
Add optional fields the AI path uses (always present, cheap, unused when OFF):
```cpp
    float sigma = 50.0f;  // AI denoise sigma (1-150); unused by Gaussian/Telea/NS
```

### 6.7 Edit: `src/core/inpaint.cpp`
The free `wmr::inpaint_residual` does **not** need an AI case here — the engine dispatches AI before calling this function (cleaner: keeps ncnn out of `inpaint.cpp`). So `inpaint.cpp` is **unchanged**. (Decision: engine-level dispatch, not inpaint-level, to keep the AI include boundary tight — `inpaint.cpp` stays ncnn-free.)

### 6.8 Edit: `src/core/watermark_engine.hpp`
Under `#ifdef WMR_AI_DENOISE`: `#include "core/ai_denoise.hpp"` (guarded); add `std::unique_ptr<NcnnDenoiser> denoiser_;` field (guarded); add `NcnnDenoiser& denoiser();` lazy accessor declaration (guarded, returns ref to the process-wide instance). No change to existing public signatures.

### 6.9 Edit: `src/core/watermark_engine.cpp`
- `#include "core/ai_denoise.hpp"` guarded at top.
- Lazy singleton accessor (guarded): function-local static `NcnnDenoiser` constructed once (R5). `initialize()` called on first use; failures logged, `is_ready()` false → Gaussian.
- In `remove_watermark_detected`, **after** `remove_watermark_alpha_blend` and **before** the existing `inpaint_residual` call, insert a guarded dispatch:
  ```cpp
  #ifdef WMR_AI_DENOISE
  if (icfg.method == InpaintMethod::AiDenoise) {
      auto& d = denoiser();
      if (d.is_ready()) {
          d.denoise(image, detection.region, alpha_map,
                    icfg.sigma, icfg.strength, icfg.padding);
          return;  // AI did the cleanup; skip Gaussian
      }
      spdlog::warn("AI denoise unavailable, falling back to Gaussian");
      icfg.method = InpaintMethod::Gaussian;
  }
  #endif
  inpaint_residual(image, detection.region, alpha_map, icfg);
  ```
  (Note: `remove_watermark_detected` currently builds `InpaintConfig icfg` and calls `inpaint_residual`; minimal change routes through the dispatch. The member `inpaint_residual` resolves alpha — reuse it or pass alpha_map directly. Implementer: thread `alpha_map` (already resolved above) into both paths to avoid the region-size heuristic.)

### 6.10 Edit: `src/cli/cli_app.hpp`
Add to `CliOptions` (always present; unused when OFF):
```cpp
std::string denoise_method;      // --denoise {ai|ns|telea|soft|off}
float denoise_sigma = 50.0f;     // --sigma 1-150
float denoise_strength_pct = 120.0f;  // --strength 0-300 (percent)
int denoise_radius = 10;         // --radius 1-25
```

### 6.11 Edit: `src/cli/cli_app.cpp`
- `#include "core/ai_denoise.hpp"` guarded.
- On `remove_cmd`/`visible_cmd`: register `--denoise/--sigma/--strength/--radius` **only under `#ifdef WMR_AI_DENOISE`** (so the OFF build has no such options — graceful non-AI). When OFF, none registered, `inpaint_strength` (existing) remains the only knob and defaults drive Gaussian.
- Resolve `InpaintConfig` from `opts`: method from `denoise_method` string (`soft`→Gaussian, `ns`→NavierStrokes, `telea`→Telea, `off`→skip cleanup, `ai`→AiDenoise [guarded]); `sigma`=`denoise_sigma`; `strength`=`denoise_strength_pct/100.0f`; `radius`=`denoise_radius`; `padding`=32.
- Pass the resolved `InpaintConfig` into `engine.remove_watermark_detected` (extend the call to forward `icfg` instead of just `inpaint_strength` — see signature note below).
- Validation: `--denoise ai` on a non-AI build → `spdlog::error(...)` + exit 2 (the option won't even exist if OFF, so this only triggers if a user hardcodes `ai` somehow; the cleaner path is the option simply isn't registered).

> **Signature note:** `remove_watermark_detected` currently takes `float inpaint_strength`. To forward the full `InpaintConfig`, EITHER (a) add an overload taking `const InpaintConfig&` (preferred — keeps the float overload for callers like video that only need strength), OR (b) add trailing defaulted params. Pick (a): add `remove_watermark_detected(image, detection, const InpaintConfig& cfg, const cv::Mat* custom_alpha=nullptr)`; keep the float overload as a forwarder that builds a default `InpaintConfig{strength}`. CLI + batch use the new overload.

### 6.12 Edit: `src/cli/batch_processor.cpp`
Mirror the CLI resolution: build `InpaintConfig` from `opts` and call the new overload. Forward the same fields. (AI flag guards identical to cli_app.cpp.)

### 6.13 Edit: `CMakeLists.txt`
- `project(wmr VERSION 1.3.0 ...)`.
- **Placement:** add `option(WMR_BUILD_AI_DENOISE ...)` near the TOP of the file (right after `project(...)` and the `CMAKE_CXX_STANDARD` lines), BEFORE `add_executable(${PROJECT_NAME} ${SOURCES})` (currently at line 30). This lets the `if(WMR_BUILD_AI_DENOISE)` block use `target_sources(${PROJECT_NAME} PRIVATE ...)` cleanly. (The existing `option(WMR_BUILD_TESTS ...)` sits at line 142 and only gates `add_subdirectory(tests)`, so it is fine where it is — but the AI option must be earlier.)
  ```cmake
  option(WMR_BUILD_AI_DENOISE "Build FDnCNN AI denoise (NCNN+Vulkan; needs vcpkg ai-denoise feature + NCNN submodule)" OFF)
  ```
- Inside `if(WMR_BUILD_AI_DENOISE)`:
  ```cmake
    # NCNN source (git submodule at the upstream-matching path)
    set(NCNN_ROOT "${CMAKE_SOURCE_DIR}/external/ncnn/ncnn-20260113-src")
    if(NOT EXISTS "${NCNN_ROOT}/CMakeLists.txt")
        message(FATAL_ERROR "WMR_BUILD_AI_DENOISE=ON requires the NCNN submodule at "
          "${NCNN_ROOT}. Run: git submodule update --init --recursive")
    endif()

    # volk (Vulkan meta-loader) — from vcpkg via the ai-denoise manifest feature.
    # find_package(Vulkan QUIET) is a diagnostic only; volk's transitive vulkan
    # headers normally satisfy the build (matches upstream CI, which installs no SDK).
    find_package(volk CONFIG REQUIRED)
    find_package(Vulkan QUIET)
    if(NOT Vulkan_FOUND)
        message(WARNING "Vulkan headers not found via find_package; relying on volk's "
          "transitive headers (upstream path). For local system-Homebrew builds install "
          "the Vulkan SDK: brew install --cask vulkan-sdk")
    endif()

    # NCNN — static, Vulkan, FDnCNN minimal layer set, no tools/tests (faithful port)
    set(NCNN_VULKAN ON CACHE BOOL "" FORCE)
    set(NCNN_VULKAN_ONLINE_SPIRV OFF CACHE BOOL "" FORCE)
    set(NCNN_SHARED_LIB OFF CACHE BOOL "" FORCE)
    set(NCNN_OPENMP OFF CACHE BOOL "" FORCE)
    set(NCNN_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(NCNN_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
    set(NCNN_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(NCNN_BENCHMARK OFF CACHE BOOL "" FORCE)
    set(NCNN_BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)
    set(NCNN_INSTALL_SDK OFF CACHE BOOL "" FORCE)
    set(NCNN_ENABLE_LTO ON CACHE BOOL "" FORCE)
    foreach(L IN ITEMS convolution relu input concat split packing cast flatten padding binaryop)
        set(WITH_LAYER_${L} ON CACHE BOOL "" FORCE)
    endforeach()
    if(MSVC)
        set(NCNN_DISABLE_RTTI OFF CACHE BOOL "" FORCE)
        set(NCNN_DISABLE_EXCEPTION OFF CACHE BOOL "" FORCE)
    endif()
    add_subdirectory(${NCNN_ROOT} ncnn)

    list(APPEND SOURCES
        ${CMAKE_SOURCE_DIR}/src/core/ai_denoise.cpp
        ${CMAKE_SOURCE_DIR}/src/core/ai_denoise_model.cpp)

    target_link_libraries(${PROJECT_NAME} PRIVATE ncnn volk::volk)
    target_compile_definitions(${PROJECT_NAME} PRIVATE WMR_AI_DENOISE=1)
  ```
  > If the `if(WMR_BUILD_AI_DENOISE)` block is placed AFTER `add_executable` (the existing `SOURCES` is consumed at line 30), use `target_sources(${PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/src/core/ai_denoise.cpp ${CMAKE_SOURCE_DIR}/src/core/ai_denoise_model.cpp)` instead of `list(APPEND SOURCES ...)`. Either works; `target_sources` is order-independent and preferred.
  > The `foreach` minimal-layer set + cache knobs mirror upstream `CMakeLists.txt` exactly (verified), keeping the NCNN build small/fast. If a chosen NCNN tag lacks some `WITH_LAYER_*` knobs, fall back to NOT setting the minimal set (build all layers) — note in the commit.
  > volk target: vcpkg's volk port exposes `volk::volk` (verified via `find_package(volk CONFIG REQUIRED)` upstream). If the installed volk version exposes a different target name, adjust the `target_link_libraries` line; do NOT fall back to a vendored/submodule volk (R8).
  > The AI sources are APPENDed to `SOURCES` (already used by `add_executable` above) — ensure the `list(APPEND SOURCES ...)` runs BEFORE the existing `add_executable(${PROJECT_NAME} ${SOURCES})` line, or use `target_sources(${PROJECT_NAME} PRIVATE ...)`. **Prefer `target_sources`** to avoid ordering coupling.

### 6.14 Edit: `tests/CMakeLists.txt`
Mirror: inside `if(WMR_BUILD_AI_DENOISE)` (the flag is visible to the tests subdir via cache), `list(APPEND LIB_SOURCES ${CMAKE_SOURCE_DIR}/src/core/ai_denoise.cpp ${CMAKE_SOURCE_DIR}/src/core/ai_denoise_model.cpp)`, `list(APPEND TEST_SOURCES unit/ai_denoise_test.cpp)`, `target_link_libraries(wmr_tests PRIVATE ncnn volk::volk)`, `target_compile_definitions(wmr_tests PRIVATE WMR_AI_DENOISE=1)`. (volk via the vcpkg `ai-denoise` feature, same as the main target — R8.) When OFF: neither source nor the test file is added (clean). NOTE: the test target must also be built with `VCPKG_MANIFEST_FEATURES=ai-denoise` when WMR_BUILD_AI_DENOISE=ON (the developer's `ctest` invocation and the §9 ON-build sequence must pass this).

### 6.15 New: `tests/unit/ai_denoise_test.cpp`
`#include <catch2/catch_test_macros.hpp>`, `#include "core/ai_denoise.hpp"`, opencv. Four `TEST_CASE(..., "[aidenoise]")` (M9). File guarded at the top with `#ifdef WMR_AI_DENOISE` (defensive — also excluded from `TEST_SOURCES` when OFF).

### 6.16 Edit: `scripts/build.sh`
After the deps check, add:
```bash
if [ "${WMR_AI_DENOISE:-0}" = "1" ]; then
  echo ">> AI Denoise mode: initializing submodules + checking Vulkan SDK"
  git submodule update --init --recursive
  if ! brew list --cask >/dev/null 2>&1 | grep -qx vulkan-sdk 2>/dev/null \
     && ! brew list --formula >/dev/null 2>&1 | grep -qx vulkan-headers 2>/dev/null; then
    # Robust check: brew list --cask can warn; verify the loader lib exists
    if [ ! -f "/usr/local/lib/libvulkan.dylib" ] && [ ! -f "/opt/homebrew/lib/libvulkan.dylib" ] \
       && [ ! -f "$(brew --prefix)/lib/libvulkan.dylib" ]; then
      echo "error: WMR_AI_DENOISE=1 needs the Vulkan SDK. Install with:" >&2
      echo "       brew install --cask vulkan-sdk" >&2
      exit 1
    fi
  fi
  CMAKE_AI_FLAG="-DWMR_BUILD_AI_DENOISE=ON"
else
  CMAKE_AI_FLAG=""
fi
```
Append `${CMAKE_AI_FLAG}` to the `cmake -S . -B ...` configure line.

### 6.17 Edit: `.github/workflows/release.yml`
Add a new job `ai-denoise` (sibling of `build`) that BUILDS AND UPLOADS the AI artifact (O1 = YES), and wire it into the existing `release` job. The 4-job `build` matrix is **unchanged** (no AI flag, no submodules, no `ai-denoise` feature — R7).

**New `ai-denoise` job (macOS arm64 only for the first AI release — mirrors upstream's verified chain, NO Vulkan SDK install):**
```yaml
  ai-denoise:
    runs-on: macos-14
    # Run on tag pushes (for releases) + manual dispatch (for testing heavy job off-cycle).
    # Skip on ordinary PR/branch pushes to avoid burning CI on a ~15+ min NCNN build.
    if: startsWith(github.ref, 'refs/tags/v') || github.event_name == 'workflow_dispatch'

    steps:
      - uses: actions/checkout@v4
        with:
          submodules: recursive          # fetches NCNN submodule (absent in the lean build jobs)

      - uses: lukka/run-vcpkg@v11
        with:
          runVcpkgInstall: true
          vcpkgJsonGlob: '**/vcpkg.json'

      - name: Configure (AI)
        run: |
          cmake -B build -S . -GNinja \
            -DCMAKE_BUILD_TYPE=${{ env.BUILD_TYPE }} \
            -DWMR_BUILD_TESTS=OFF \
            -DWMR_BUILD_AI_DENOISE=ON \
            -DVCPKG_MANIFEST_FEATURES="ai-denoise" \
            -DCMAKE_TOOLCHAIN_FILE=${{ github.workspace }}/vcpkg/scripts/buildsystems/vcpkg.cmake

      - name: Build
        run: cmake --build build --config ${{ env.BUILD_TYPE }}

      - name: Smoke test (--version)
        run: ./build/wmr --version

      - name: Package
        run: |
          mkdir -p dist
          cp build/wmr dist/wmr-macos-arm64-ai
          chmod +x dist/wmr-macos-arm64-ai
          cp LICENSE-AI.md dist/           # licenses travel with the shipped binary

      - uses: actions/upload-artifact@v4
        with:
          name: wmr-macos-arm64-ai
          path: dist/
```

**Existing `release` job — two edits:**
1. Change `needs: build` → `needs: [build, ai-denoise]` so the release waits for the AI artifact too. (Keep `if: startsWith(github.ref, 'refs/tags/v')` — the AI job's own `if` means on a tag push both run; on a non-tag `workflow_dispatch` the release job is skipped, which is fine.)
2. Add the AI artifact to the `softprops/action-gh-release@v2` `files:` list:
```yaml
      files: |
        artifacts/wmr-macos-arm64/wmr-macos-arm64
        artifacts/wmr-macos-x86_64/wmr-macos-x86_64
        artifacts/wmr-linux-x86_64/wmr-linux-x86_64
        artifacts/wmr-windows-x86_64.exe/wmr-windows-x86_64.exe
        artifacts/wmr-macos-arm64-ai/wmr-macos-arm64-ai
        artifacts/wmr-macos-arm64-ai/LICENSE-AI.md
```
> The `release` job already has `permissions: contents: write` (verified), so no permission change is needed. The existing `download-artifact@v4` with `path: artifacts` already pulls EVERY uploaded artifact (it has no `name:` filter), so the new AI artifact is downloaded automatically — only the `files:` list needs the two new lines. The AI artifact name `wmr-macos-arm64-ai` is distinct from `wmr-macos-arm64`, so there is no overwrite collision (R7).

### 6.18 Edits: `CLAUDE.md`, `README.md`, `CHANGELOG.md`, `vcpkg.json`, `LICENSE-AI.md`
- **CLAUDE.md** — new section under "## Pipeline": AI Denoise as an optional cleanup; the `WMR_BUILD_AI_DENOISE` flag; the Inpaint guard pattern (`#ifdef WMR_AI_DENOISE`); that the OFF build is the lean default; that AI needs the NCNN git submodule + the vcpkg `ai-denoise` feature (volk) — NOT a standalone Vulkan SDK install in CI. Note the separate `ai-denoise` CI job that ships `wmr-macos-arm64-ai`.
- **README.md** — new "AI Denoise (optional)" subsection (build flag, NCNN submodule + vcpkg `ai-denoise` feature, `--denoise/--sigma/--strength/--radius`, GPU-vs-CPU). **New "Release binaries" note** explaining the two macOS variants: `wmr-macos-arm64` (lean, no AI) and `wmr-macos-arm64-ai` (AI-enabled; see `LICENSE-AI.md`). "Source Projects" += KAIR (FDnCNN, MIT), Tencent/ncnn (BSD-3-Clause), zeux/volk (MIT).
- **CHANGELOG.md** — new `## [1.3.0] - 2026-06-28` section: Added (FDnCNN AI denoise; `wmr-macos-arm64-ai` release variant; `LICENSE-AI.md`); Changed (default cleanup when built with AI); Build (`WMR_BUILD_AI_DENOISE`, NCNN submodule, vcpkg `ai-denoise` feature, CI `ai-denoise` job); note the **~6.8 MB embedded-model-header commit** (O3) and the model-provenance path.
- **vcpkg.json** — `"version": "1.3.0"`; add the new `ai-denoise` feature declaring `volk` (M1). The default `dependencies` array is UNCHANGED (no volk/ncnn in default features → lean binaries stay AI-free, R7).
- **CMakeLists.txt** — version `1.3.0` (already in §6.13).
- **New `LICENSE-AI.md`** (repo root) — the third-party license bundle for the AI binary (NCNN BSD-3-Clause + volk MIT + KAIR/FDnCNN MIT + allenk conversion credit). Shipped inside the AI artifact (§6.17 copies it next to the binary).

---

## 7. Model Provenance (explicit)

We do NOT re-convert FDnCNN from KAIR (no PyTorch/onnx/ncnn toolchain assumed). The converted NCNN model is upstream's work (MIT). Concretely:

1. Build `allenk/GeminiWatermarkTool` once with `ENABLE_AI_DENOISE=ON` (its CI/build runs `ncnn2mem`, emitting `external/ncnn/model-convert/output/model_core.id.h` + `model_core.mem.h`).
2. Copy those two generated files into our `assets/`.
3. Commit them with the attribution header (FDnCNN © cszn/KAIR MIT; NCNN conversion © allenk/GeminiWatermarkTool MIT).
4. Record the upstream commit SHA used + the provenance path in CHANGELOG `[1.3.0]` and the `ai_denoise_model.cpp` header comment.

If step 1 is impractical, alternatives (M3 items 2–3). The bytes must be upstream's converted weights — do NOT invent or hand-derive them.

---

## 8. Phased Build Sequence

1. **NCNN submodule + vcpkg `ai-denoise` feature + provenance + CMake flag (no code yet).** Add the NCNN submodule; add the `ai-denoise` feature (volk) to `vcpkg.json`; capture model headers into `assets/`; add the `option` + `if(WMR_BUILD_AI_DENOISE)` block; confirm `cmake -DWMR_BUILD_AI_DENOISE=ON -DVCPKG_MANIFEST_FEATURES=ai-denoise` configures (NCNN+volk found) AND `=OFF` (no feature) configures unchanged. Land a skeleton `ai_denoise.cpp` that just `initialize()`s and logs the device.
2. **NcnnDenoiser full port.** Port `ai_denoise.cpp`/`ncnn_shim.hpp`/`ai_denoise_model.cpp`; verify model loads (`is_ready()` true) on CPU; verify GPU path on a Vulkan-capable machine.
3. **Engine + inpaint wiring.** Guarded `AiDenoise` enum value; engine lazy denoiser + dispatch + Gaussian fallback. Default-when-built verified.
4. **CLI + batch.** `--denoise/--sigma/--strength/--radius`; `InpaintConfig` overload; non-AI build graceful.
5. **Tests + docs + CI + version.** `[aidenoise]` tests; CLAUDE/README/CHANGELOG; optional CI job; bump 1.3.0.
6. **Verification (§9) + rollback check.**

---

## 9. Build & Verification Sequence

```bash
# ---- A. OFF build (must be identical to today) ----
cmake -B build -S . -GNinja \
  -DOpenCV_DIR=$(brew --prefix opencv)/lib/cmake/opencv4 \
  -DFFTW3f_DIR=$(brew --prefix fftw)/lib/cmake/fftw3 \
  -DWMR_BUILD_TESTS=ON
cmake --build build
ctest --test-dir build --output-on-failure      # full suite green; no [aidenoise] tag present
./build/wmr --help                                # no --denoise/--sigma/--strength/--radius

# Confirm OFF build has no AI sources/defs (regression guard):
ninja -C build -t commands | grep -c ai_denoise   # -> 0
grep -c WMR_AI_DENOISE build/compile_commands.json # -> 0

# ---- B. ON build (developer; needs NCNN submodule + volk) ----
# Two local paths:
#   B1. vcpkg path (matches CI — recommended for parity):
git submodule update --init --recursive
cmake -B build-ai -S . -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DWMR_BUILD_TESTS=ON -DWMR_BUILD_AI_DENOISE=ON \
  -DVCPKG_MANIFEST_FEATURES="ai-denoise" \
  -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake
cmake --build build-ai
#   B2. System-Homebrew path (scripts/build.sh WMR_AI_DENOISE=1): brew installs vulkan-sdk
#       for headers/loader; configure adds -DWMR_BUILD_AI_DENOISE=ON (M5).

# New AI tag (runs on CPU at minimum):
./build-ai/wmr_tests "[aidenoise]"

# Full suite still green (video/inpaint/blend unaffected):
ctest --test-dir build-ai --output-on-failure

# Manual still before/after:
./build-ai/wmr remove <gemini.png> --denoise ai  -o out-ai.png
./build-ai/wmr remove <gemini.png> --denoise soft -o out-soft.png
./build-ai/wmr remove <gemini.png> --denoise off  -o out-none.png   # no cleanup
./build-ai/wmr remove <gemini.png> --sigma 75 --strength 150 -o out-aggr.png

# Fallback: force GPU off (e.g. unset Vulkan loader) → confirm Gaussian path runs:
# (unset VK_ICD_FILENAMES / move libMoltenVK aside temporarily, then:)
./build-ai/wmr remove <gemini.png> --denoise ai -o out-fallback.png  # logs "AI unavailable, falling back to Gaussian"
```

**Acceptance criteria:**
- OFF build: `ai_denoise*` source count = 0; `WMR_AI_DENOISE` define count = 0; full `ctest` green; `wmr --help` shows no `--denoise/--sigma/--strength/--radius`.
- ON build: `[aidenoise]` tests pass (model loads on CPU; synthetic residual repaired; background ≤ tolerance unchanged; forced-failure → Gaussian, removal succeeds).
- ON build: full `ctest` green (no regression in `[blend]/[inpaint]/[v2]/[integration]`).
- ON build: `--denoise ai` runs and logs device (GPU name or "CPU (N threads)"); `--denoise soft` reproduces pre-AI Gaussian output; `--denoise off` skips cleanup.
- ON build, forced Vulkan failure: removal still succeeds via Gaussian (R4).
- Version reports `1.3.0`.
- **CI / release (O1):** On a `v*` tag push, the workflow produces **5 artifacts** (4 lean + `wmr-macos-arm64-ai`). The release's `files:` list includes all 5 + `LICENSE-AI.md`. **The 4 lean binaries must contain NO ncnn/volk/MoltenVK symbols** — verify post-release with:
  ```bash
  # Download wmr-macos-arm64 from the release and confirm it is AI-free:
  nm wmr-macos-arm64 2>/dev/null | grep -ci ncnn          # -> 0
  otool -L wmr-macos-arm64 | grep -iE 'moltenvk|vulkan'   # -> no match
  # The AI binary, conversely, MUST link MoltenVK/volk at runtime:
  otool -L wmr-macos-arm64-ai | grep -iE 'moltenvk|vulkan|libvulkan'  # -> match (or static)
  ```

---

## 10. Rollback Notes

- All AI code is in new files (`ai_denoise.{hpp,cpp}`, `ai_denoise_model.cpp`, `ncnn_shim.hpp`, `LICENSE-AI.md`) + `#ifdef WMR_AI_DENOISE`-guarded edits to `inpaint.hpp`, `watermark_engine.{hpp,cpp}`, `cli_app.{hpp,cpp}`, `batch_processor.cpp`, both CMakeLists, `build.sh`, `release.yml`, `vcpkg.json` (the `ai-denoise` feature).
- `git revert <commit>` + `git submodule deinit external/ncnn/ncnn-20260113-src` + `git rm external/ncnn/ncnn-20260113-src assets/model_core.* LICENSE-AI.md` + revert the `release.yml` `needs:`/`files:` edits restores the exact 1.2.0 state. (volk is a vcpkg feature dep, not a submodule — no submodule deinit needed for it; reverting the `vcpkg.json` feature removes it.) The OFF/lean build was never affected, so even a partial revert (just the version bump + docs + `release.yml`) leaves users on a working 1.2.0-equivalent.
- The pre-existing release artifacts are untouched in the sense that the 4 lean binaries build identically; only the new `wmr-macos-arm64-ai` artifact is AI-bearing. To stop shipping the AI artifact without a full revert, simply set `if: false` on the `ai-denoise` job and drop its two `files:` lines from the `release` job — a one-commit hotfix that leaves the source and the 4 lean binaries intact.

---

## 11. Resolved Decisions (O1–O4 — maintainer-confirmed, integrated above)

- **O1 — Ship an AI release artifact: YES (DECIDED).** The dedicated `ai-denoise` CI job BUILDS AND UPLOADS `wmr-macos-arm64-ai` (macOS arm64 only for the first AI release; x86_64/Linux/Windows AI builds are future work). Gated on `v*` tags + `workflow_dispatch`. Wired into the `release` job (`needs: [build, ai-denoise]` + the artifact + `LICENSE-AI.md` added to `files:`). The 4 lean default binaries stay AI-free (R7). CI chain mirrors upstream's verified `build.yml` (submodules for NCNN + vcpkg `ai-denoise` feature for volk + NO Vulkan SDK install). See §6.17.
- **O2 — NCNN tag: DECIDED.** Pin the nearest real `Tencent/ncnn` release tag at execution time (target `20260113` to match upstream; fall back to the nearest real release tag if absent). Record the SHA in `.gitmodules` + CHANGELOG. The exact tag is a build detail, not a design decision.
- **O3 — Commit the model headers: YES (DECIDED, definitively).** Commit `assets/model_core.{mem.h,id.h}` (~6.8 MB). No "revisit if clone times suffer" hedge. Repo-size impact noted in CHANGELOG. Required so the AI CI job (O1) builds from a clean checkout without a network fetch. See §6.5, §6.18, M3.
- **O4 — Dispatch location: engine-level (DECIDED).** AI dispatches in `WatermarkEngine::remove_watermark_detected`, NOT in the free `wmr::inpaint_residual` (keeps ncnn out of `inpaint.cpp`). See §6.7, §6.9.
