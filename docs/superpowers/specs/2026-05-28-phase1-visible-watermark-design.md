# Phase 1: Visible Watermark Removal — Design Spec

## Context

The project needs a working foundation: C++20 build system, core engine for removing Gemini's visible sparkle-logo watermark, and a CLI to process single images. This phase establishes the architecture that SynthID removal and video support will later plug into.

## Scope

- CMake build system with vcpkg dependency management
- Reverse alpha blending engine using calibrated 48x48 and 96x96 watermark masks
- Single-image CLI processing (in-place or to output path)
- Test against `test-images/2400x1792-gemini.png`

**Out of scope for Phase 1:** Detection (auto-skip), batch processing, inpainting/AI denoise, SynthID removal, video support, GUI.

## Architecture

Single executable, no library split yet. Namespace `wmr`. Core logic separated from CLI.

```
src/main.cpp                  → Entry point, dispatches to CLI
src/core/types.hpp            → Enums, result structs, position rules
src/core/blend_modes.hpp/cpp  → Stateless alpha blend math
src/core/watermark_engine.hpp/cpp → Engine: owns alpha maps, orchestrates removal
src/cli/cli_app.hpp/cpp       → CLI11 argument parsing and processing loop
assets/embedded_assets.hpp    → constexpr PNG byte arrays (48x48, 96x96 masks)
```

Build config:
```
CMakeLists.txt
CMakePresets.json
vcpkg.json
```

## Components

### types.hpp

Foundation types shared across all modules.

```cpp
namespace wmr {

enum class [[nodiscard]] ResultCode {
    Success, FileNotFound, InvalidFormat, ProcessingFailed, SaveFailed
};

enum class WatermarkSize { Small, Large };

struct ProcessResult {
    bool success = false;
    bool skipped = false;
    float confidence = 0.0f;
    std::string message;
};

WatermarkSize get_watermark_size(int width, int height);
// <= 1024 on either dimension → Small (48x48)
// > 1024 on both → Large (96x96)

struct WatermarkPosition {
    int margin_right;
    int margin_bottom;
    int logo_size;
    cv::Point get_position(int image_width, int image_height) const;
};

WatermarkPosition get_watermark_config(int width, int height);
// Small: 32px margins, 48px logo
// Large: 64px margins, 96px logo

}
```

### blend_modes.hpp/cpp

Stateless free functions for alpha blending math.

```cpp
namespace wmr {

cv::Mat calculate_alpha_map(const cv::Mat& bg_capture);
// Derives alpha from max(R,G,B) / 255.0

void remove_watermark_alpha_blend(
    cv::Mat& image, const cv::Mat& alpha_map,
    const cv::Point& position, float logo_value = 255.0f);
// original = (watermarked - alpha * logo) / (1 - alpha)
// alpha_threshold = 0.002f (skip negligible)
// max_alpha = 0.99f (avoid division instability)

void add_watermark_alpha_blend(
    cv::Mat& image, const cv::Mat& alpha_map,
    const cv::Point& position, float logo_value = 255.0f);
// watermarked = alpha * logo + (1 - alpha) * original

}
```

### watermark_engine.hpp/cpp

Owns the alpha maps and orchestrates removal.

```cpp
namespace wmr {

class WatermarkEngine {
public:
    WatermarkEngine();
    // Decodes embedded PNGs via cv::imdecode, derives alpha maps

    void remove_watermark(cv::Mat& image,
                          std::optional<WatermarkSize> force_size = std::nullopt);
    void add_watermark(cv::Mat& image,
                       std::optional<WatermarkSize> force_size = std::nullopt);

    const cv::Mat& get_alpha_map(WatermarkSize size) const;

private:
    cv::Mat alpha_map_small_;   // 48x48, CV_32FC1
    cv::Mat alpha_map_large_;   // 96x96, CV_32FC1
    float logo_value_ = 255.0f;

    cv::Mat create_interpolated_alpha(int width, int height, WatermarkSize size) const;
    void init_alpha_maps();
};

}
```

Flow: `remove_watermark` → auto-detect size → get position config → interpolate alpha map to match region → call `remove_watermark_alpha_blend`.

### embedded_assets.hpp

Two calibrated PNG captures from GeminiWatermarkTool (MIT licensed), stored as `inline constexpr` byte arrays.

```cpp
namespace wmr::embedded {

inline constexpr unsigned char bg_48_png[] = { /* ~1677 bytes */ };
inline constexpr size_t bg_48_png_size = 1677;

inline constexpr unsigned char bg_96_png[] = { /* ~8165 bytes */ };
inline constexpr size_t bg_96_png_size = 8165;

}
```

At runtime, `WatermarkEngine` decodes these via `cv::imdecode` and passes to `calculate_alpha_map`.

### cli_app.hpp/cpp

```cpp
namespace wmr {

struct CliOptions {
    std::string input_path;
    std::string output_path;
    bool force = false;
    bool force_small = false;
    bool force_large = false;
    bool verbose = false;
};

int run_cli(int argc, char* argv[]);

}
```

CLI surface:
```
wmr [OPTIONS] input
  -o, --output PATH       Output file (default: overwrite in-place)
  -f, --force             Skip detection (process unconditionally)
      --force-small       Force 48x48 watermark
      --force-large       Force 96x96 watermark
  -v, --verbose           Verbose output
  -V, --version           Version
  -h, --help              Help
```

Processing loop: parse args → create WatermarkEngine → load image → remove_watermark → save result.

## Build System

### CMakeLists.txt

- CMake 3.21+, C++20 required
- Project `wmr` version `0.1.0`
- Single executable, Ninja generator
- Include dirs: `src/` and `assets/`
- Compile defs: `APP_VERSION`, `APP_NAME`
- Link: OpenCV, fmt::fmt, CLI11::CLI11, spdlog::spdlog

Platform flags:
- macOS: `-Wall -Wextra -Wpedantic`, `-O3` release, universal binary (`x86_64;arm64`), deployment target 11.0
- Linux: `-Wall -Wextra -Wpedantic`, `-O3` release, `-static-libgcc -static-libstdc++`
- Windows (MSVC): `/W4 /utf-8`, `/O2` release, `/MT` static CRT

### CMakePresets.json

Inheritance: `base` → `vcpkg-base` → `mac-base` → `mac-x64-Release`

macOS preset:
- Generator: Ninja
- Toolchain: `$env{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake`
- Architecture: `x86_64;arm64`
- Deployment target: 11.0

### vcpkg.json

```json
{
  "name": "wmr",
  "version": "0.1.0",
  "dependencies": [
    { "name": "opencv4", "default-features": false, "features": ["jpeg", "png", "webp"] },
    "fmt",
    "cli11",
    "spdlog"
  ]
}
```

No features or overrides — just core deps for Phase 1.

## Verification

1. Configure and build on macOS with vcpkg + Ninja
2. Run `wmr test-images/2400x1792-gemini.png -o test-output.png`
3. Open output image — bottom-right watermark area should be cleaned
4. Run `wmr --help` — shows usage text
5. Run `wmr --version` — shows `0.1.0`
