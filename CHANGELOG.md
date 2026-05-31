# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Phase 6: Video Watermark Removal + CLI Polish — COMPLETE

#### Added
- `VideoReader` class (`src/video/video_reader.hpp/cpp`) — FFmpeg demux+decode pipeline with seeking and frame counting
- `VideoWriter` class (`src/video/video_writer.hpp/cpp`) — FFmpeg encode pipeline (libx264, CRF 14, High profile) with audio passthrough:
  - Audio streams created before MP4 header write (valid moov atom)
  - Fresh input context for audio packet copy with timestamp rescaling
  - BGR24→YUV420P colorspace conversion via swscale
- `VideoProcessor` class (`src/video/video_processor.hpp/cpp`) — frame-by-frame watermark removal:
  - Shot-level NCC detection: samples frames across first 90% of video, takes median position/confidence
  - Per-frame occlusion gate: skips frames where watermark not reliably detected
  - Position refinement: falls back to shot anchor if detection drifts beyond tolerance
  - Frame dimension guard against corrupt seek artifacts
  - Progress output with fps and ETA
- `video` CLI subcommand: `wmr video input.mp4 -o output.mp4` with `--legacy`, `--variant`, `--crf`, `--preset`, `--codec`, `--force` options
- Codebook-free SynthID removal via `NoiseResidualSubtractor` (`src/synthid/noise_residual_subtractor.hpp/cpp`):
  - Estimates carrier from bilateral filter noise residual
  - `--codebook-free` flag as alternative to `--codebook`
- `cmake/FindFFMPEG.cmake` — cross-platform FFmpeg discovery via pkg-config / FFMPEG_ROOT (covers Homebrew and system package managers)
- CLI header with version, description, GitHub URL, and copyright
- Contextual subcommand help: shows subcommand-specific help when required args are missing (e.g. `wmr video`)
- Version output includes GitHub URL

#### Changed
- CMake links FFmpeg via imported targets (`FFMPEG::avformat` etc.) instead of variable-based linking
- `vcpkg.json` version bumped to 0.2.0
- Project version bumped to 0.2.0

### Phase 5: Unified CLI + Test Suite — COMPLETE

#### Added
- CLI restructured with CLI11 subcommands: `remove`, `detect`, `visible`, `synthid`, `build-codebook`
- Batch processing module (`src/cli/batch_processor.hpp/cpp`) — directory scanning with optional `--recursive`, parallel-safe
- `BatchResult` struct tracking total/succeeded/failed/skipped counts
- Catch2 v3 test suite with 17 tests, 54 assertions (all passing):
  - Unit tests: blend_modes (3), fft_context (4), spectral_codebook (4), inpaint (3)
  - Integration tests: visible_pipeline (3, 2 SKIP in build dir, pass from project root)
- Tests linked via `tests/CMakeLists.txt` with `catch_discover_tests()` and CTest integration
- Build option `WMR_BUILD_TESTS` (default ON) to toggle test building

### Phase 4: SynthID Detection + Codebook Builder — COMPLETE

#### Added
- `SynthidDetector` class (`src/detection/synthid_detector.hpp/cpp`) — 4-method Bayesian detection:
  - Noise correlation via bilateral filter denoise + NCC of noise spectrum vs profile
  - Carrier phase matching via element-wise cosine of phase differences
  - Structure ratio: energy at carrier bins vs total
  - Multi-scale consistency: phase coherence at 1x, 0.5x, 0.25x scales
  - Weighted fusion: noise_corr×0.35 + carrier_phase×0.35 + structure×0.15 + multi_scale×0.15
  - Configurable threshold (default 0.50)
- `CodebookBuilder` class (`src/synthid/codebook_builder.hpp/cpp`) — build spectral codebooks from reference images:
  - Per-channel FFT magnitude/phase accumulation
  - Automatic resolution bucketing
  - Consistency computation (std dev across samples)
  - Quality gate: warns on <3 samples per resolution

### Phase 3: SynthID Spectral Infrastructure — COMPLETE

#### Added
- `FftContext` class (`src/core/fft_context.hpp/cpp`) — FFTW3 wrapper with plan caching:
  - Forward/inverse 2D FFT with CV_32FC1/CV_32FC2 interop
  - Magnitude, phase, and polar reconstruction utilities
  - Plan caching keyed on (rows, cols, direction) with dummy arrays for FFTW_MEASURE safety
- `SpectralProfile` struct in `src/core/types.hpp` — per-resolution FFT data (magnitude, phase, consistency per BGR channel)
- `SpectralCodebook` class (`src/synthid/spectral_codebook.hpp/cpp`) — JSON-based codebook persistence:
  - Save/load with nearest-resolution fallback
  - `--codebook` CLI flag for specifying codebook path
- `CodebookSubtractor` class (`src/synthid/codebook_subtractor.hpp/cpp`) — multi-pass spectral subtraction:
  - Aggressive→moderate→gentle removal schedule
  - `--synthid-strength` CLI flag (0.0–2.0, default 1.0)
- `fftw3f` linked via vcpkg, FFTW3::fftw3f CMake target

#### Fixed
- SpectralProfile aggregate init bug: explicit field assignment prevents misaligned brace initialization
- FFTW_MEASURE input corruption: dummy arrays prevent plan creation from overwriting real data
- OpenCV `cv::cos` (nonexistent) replaced with element-wise `std::cos` loop in carrier phase matching

### Phase 2: Visible Watermark Detection + Inpainting — COMPLETE

#### Added
- `DetectionResult` struct in `src/core/types.hpp` — detection result with confidence scores, region, and per-stage scores
- `NccDetector` class (`src/detection/ncc_detector.hpp/cpp`) — three-stage NCC detection pipeline:
  - Stage 1: Spatial NCC via template matching with circuit breaker at 0.25
  - Stage 2: Gradient NCC via Sobel-filtered magnitude matching
  - Stage 3: Variance analysis comparing watermark region to reference region
  - Heuristic fusion: spatial×0.50 + gradient×0.30 + variance×0.20
  - Detection threshold: confidence ≥ 0.35
- `InpaintMethod` enum and `InpaintConfig` struct in `src/core/inpaint.hpp`
- `inpaint_residual()` function (`src/core/inpaint.hpp/cpp`) — three traditional inpainting methods:
  - Gaussian soft inpaint with gradient-weighted mask
  - TELEA inpaint (cv::inpaint INPAINT_TELEA)
  - Navier-Stokes inpaint (cv::inpaint INPAINT_NS)
  - Configurable strength, radius, and padding
- `opencv_photo` linked for cv::inpaint support
- WatermarkEngine integration: `detect_watermark()`, `remove_watermark_detected()`, `inpaint_residual()` methods
- CLI updated: default flow is now detect→remove→inpaint; `--force` skips detection; `--detect-only` reports results
- Verified: 80% detection confidence on test image, successful removal with Gaussian inpainting (9671 active pixels)

## [0.1.0] - 2026-05-28

### Phase 1: Visible Watermark Removal

#### Added
- C++20 project structure with CMake build system (CMakePresets.json, vcpkg.json)
- `WatermarkEngine` class — reverse alpha blending for visible watermark removal
- `blend_modes` module — alpha map calculation and forward/reverse alpha blending
- Embedded PNG assets (48x48 and 96x96 background captures) in `assets/embedded_assets.hpp`
- CLI via CLI11: `wmr input.png -o output.png` with `--force`, `--force-small`, `--force-large`, `-v` flags
- Auto-detection of watermark size based on image dimensions
- Support for PNG, JPEG, WebP output formats with quality preservation
