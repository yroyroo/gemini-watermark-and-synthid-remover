# Phase 2: Visible Watermark Detection + Inpainting — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:subagent-driven-development (recommended) or superpowers-extended-cc:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add NCC-based watermark detection and traditional inpainting to make visible watermark removal robust: detect before removing, clean up residuals.

**Architecture:** Port GWT's three-stage NCC detection (spatial, gradient, variance) into a standalone `NccDetector` class. Port GWT's `inpaint_residual` into standalone `inpaint` module with three methods (Gaussian, TELEA, Navier-Stokes). WatermarkEngine integrates both — `remove_watermark` now optionally detects first and inpaints after.

**Tech Stack:** C++20, OpenCV (imgproc, photo), existing vcpkg deps (no new dependencies).

**Reference:** `allenk/GeminiWatermarkTool` — `src/core/watermark_engine.cpp` (`detect_watermark`, `guided_detect`, `inpaint_residual` methods)

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `src/core/types.hpp` | Modify | Add `DetectionResult` struct |
| `src/detection/ncc_detector.hpp` | Create | NccDetector class declaration |
| `src/detection/ncc_detector.cpp` | Create | Three-stage NCC detection |
| `src/core/inpaint.hpp` | Create | Inpaint function declarations |
| `src/core/inpaint.cpp` | Create | Gaussian, TELEA, Navier-Stokes inpaint |
| `src/core/watermark_engine.hpp` | Modify | Add detect + inpaint methods |
| `src/core/watermark_engine.cpp` | Modify | Integrate detection and inpainting |
| `src/cli/cli_app.hpp` | Modify | Add `detect_only` flag |
| `src/cli/cli_app.cpp` | Modify | Detect→remove→inpaint pipeline |
| `CMakeLists.txt` | Modify | Add new sources and include dir |

---

### Task 1: Add DetectionResult to types.hpp

**Goal:** Add the `DetectionResult` struct to shared types so all modules can use it.

**Files:**
- Modify: `src/core/types.hpp`

**Acceptance Criteria:**
- [ ] `DetectionResult` struct defined with all fields
- [ ] Existing code compiles unchanged

**Verify:** `cmake --build build/mac-x64-Release 2>&1 | tail -5` → BUILD SUCCEEDED

**Steps:**

- [ ] **Step 1: Add DetectionResult to types.hpp**

Add after the `ProcessResult` struct (after line 23):

```cpp
struct DetectionResult {
    bool detected = false;
    float confidence = 0.0f;
    cv::Rect region;
    WatermarkSize size = WatermarkSize::Small;
    float spatial_score = 0.0f;
    float gradient_score = 0.0f;
    float variance_score = 0.0f;
};
```

- [ ] **Step 2: Verify build**

Run: `cmake --build build/mac-x64-Release 2>&1 | tail -5`
Expected: BUILD SUCCEEDED (no code uses DetectionResult yet)

- [ ] **Step 3: Commit**

```bash
git add src/core/types.hpp
git commit -m "feat: add DetectionResult struct to shared types"
```

---

### Task 2: Create NccDetector module

**Goal:** Implement the three-stage NCC detection algorithm ported from GWT's `WatermarkEngine::detect_watermark`.

**Files:**
- Create: `src/detection/ncc_detector.hpp`
- Create: `src/detection/ncc_detector.cpp`
- Modify: `CMakeLists.txt`

**Acceptance Criteria:**
- [ ] NccDetector class compiles and links
- [ ] `detect()` returns DetectionResult with confidence scores
- [ ] Three stages computed: spatial NCC, gradient NCC, variance analysis
- [ ] Circuit breaker at spatial score < 0.25

**Verify:** `cmake --build build/mac-x64-Release 2>&1 | tail -5` → BUILD SUCCEEDED

**Steps:**

- [ ] **Step 1: Create `src/detection/ncc_detector.hpp`**

```cpp
#pragma once

#include <opencv2/core.hpp>
#include <optional>

#include "core/types.hpp"

namespace wmr {

class NccDetector {
public:
    NccDetector(const cv::Mat& alpha_small, const cv::Mat& alpha_large);

    DetectionResult detect(
        const cv::Mat& image,
        std::optional<WatermarkSize> force_size = std::nullopt) const;

private:
    cv::Mat alpha_map_small_;
    cv::Mat alpha_map_large_;

    const cv::Mat& get_alpha_map(WatermarkSize size) const;
};

} // namespace wmr
```

- [ ] **Step 2: Create `src/detection/ncc_detector.cpp`**

```cpp
#include "detection/ncc_detector.hpp"

#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>
#include <algorithm>

namespace wmr {

NccDetector::NccDetector(const cv::Mat& alpha_small, const cv::Mat& alpha_large)
    : alpha_map_small_(alpha_small.clone()), alpha_map_large_(alpha_large.clone()) {}

const cv::Mat& NccDetector::get_alpha_map(WatermarkSize size) const {
    return (size == WatermarkSize::Small) ? alpha_map_small_ : alpha_map_large_;
}

DetectionResult NccDetector::detect(
    const cv::Mat& image,
    std::optional<WatermarkSize> force_size) const
{
    DetectionResult result;

    if (image.empty()) return result;

    const WatermarkSize size = force_size.value_or(
        get_watermark_size(image.cols, image.rows));
    const auto config = get_watermark_config(image.cols, image.rows);
    const cv::Point pos = config.get_position(image.cols, image.rows);
    const cv::Mat& alpha_map = get_alpha_map(size);

    result.size = size;
    result.region = cv::Rect(pos.x, pos.y, alpha_map.cols, alpha_map.rows);

    // Clamp ROI to image bounds
    const int x1 = std::max(0, pos.x);
    const int y1 = std::max(0, pos.y);
    const int x2 = std::min(image.cols, pos.x + alpha_map.cols);
    const int y2 = std::min(image.rows, pos.y + alpha_map.rows);

    if (x1 >= x2 || y1 >= y2) {
        spdlog::debug("Detection: ROI out of bounds");
        return result;
    }

    // Extract region, convert to grayscale float [0,1]
    const cv::Rect image_roi(x1, y1, x2 - x1, y2 - y1);
    const cv::Mat region = image(image_roi);

    cv::Mat gray_region;
    if (region.channels() >= 3) {
        cv::cvtColor(region, gray_region, cv::COLOR_BGR2GRAY);
    } else {
        gray_region = region.clone();
    }

    cv::Mat gray_f;
    gray_region.convertTo(gray_f, CV_32F, 1.0 / 255.0);

    // Corresponding alpha region
    const cv::Rect alpha_roi(x1 - pos.x, y1 - pos.y, x2 - x1, y2 - y1);
    const cv::Mat alpha_region = alpha_map(alpha_roi);

    // Stage 1: Spatial NCC
    cv::Mat spatial_match;
    cv::matchTemplate(gray_f, alpha_region, spatial_match, cv::TM_CCOEFF_NORMED);

    double spatial_score;
    cv::minMaxLoc(spatial_match, nullptr, &spatial_score);
    result.spatial_score = static_cast<float>(spatial_score);

    // Circuit breaker
    constexpr double kSpatialThreshold = 0.25;
    if (spatial_score < kSpatialThreshold) {
        spdlog::debug("Detection: spatial={:.3f} < {:.2f}, rejected",
                      spatial_score, kSpatialThreshold);
        result.confidence = static_cast<float>(spatial_score * 0.5);
        return result;
    }

    // Stage 2: Gradient NCC
    cv::Mat img_gx, img_gy, img_gmag;
    cv::Sobel(gray_f, img_gx, CV_32F, 1, 0, 3);
    cv::Sobel(gray_f, img_gy, CV_32F, 0, 1, 3);
    cv::magnitude(img_gx, img_gy, img_gmag);

    cv::Mat alpha_gx, alpha_gy, alpha_gmag;
    cv::Sobel(alpha_region, alpha_gx, CV_32F, 1, 0, 3);
    cv::Sobel(alpha_region, alpha_gy, CV_32F, 0, 1, 3);
    cv::magnitude(alpha_gx, alpha_gy, alpha_gmag);

    cv::Mat grad_match;
    cv::matchTemplate(img_gmag, alpha_gmag, grad_match, cv::TM_CCOEFF_NORMED);

    double grad_score;
    cv::minMaxLoc(grad_match, nullptr, &grad_score);
    result.gradient_score = static_cast<float>(grad_score);

    // Stage 3: Variance analysis
    double var_score = 0.0;
    const int ref_h = std::min(y1, config.logo_size);

    if (ref_h > 8) {
        const cv::Rect ref_roi(x1, y1 - ref_h, x2 - x1, ref_h);
        const cv::Mat ref_region = image(ref_roi);
        cv::Mat gray_ref;
        if (ref_region.channels() >= 3) {
            cv::cvtColor(ref_region, gray_ref, cv::COLOR_BGR2GRAY);
        } else {
            gray_ref = ref_region;
        }

        cv::Scalar s_wm, s_ref;
        cv::meanStdDev(gray_region, cv::noArray(), s_wm);
        cv::meanStdDev(gray_ref, cv::noArray(), s_ref);

        if (s_ref[0] > 5.0) {
            var_score = std::clamp(1.0 - (s_wm[0] / s_ref[0]), 0.0, 1.0);
        }
    }
    result.variance_score = static_cast<float>(var_score);

    // Heuristic fusion
    const double confidence =
        (spatial_score * 0.50) +
        (grad_score * 0.30) +
        (var_score * 0.20);

    result.confidence = static_cast<float>(std::clamp(confidence, 0.0, 1.0));
    result.detected = (result.confidence >= 0.35f);

    spdlog::debug("Detection: spatial={:.3f}, grad={:.3f}, var={:.3f} "
                  "-> conf={:.3f} ({})",
                  spatial_score, grad_score, var_score, result.confidence,
                  result.detected ? "DETECTED" : "not detected");

    return result;
}

} // namespace wmr
```

- [ ] **Step 3: Update `CMakeLists.txt` — add detection sources and include dir**

Replace the SOURCES block and include dirs:

```cmake
set(SOURCES
    src/main.cpp
    src/core/blend_modes.cpp
    src/core/watermark_engine.cpp
    src/core/inpaint.cpp
    src/detection/ncc_detector.cpp
    src/cli/cli_app.cpp
)

add_executable(${PROJECT_NAME} ${SOURCES})

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/assets
)
```

Note: `src/core/inpaint.cpp` is added now but created in Task 3. Create an empty stub file to keep the build passing:

```cpp
// Stub — implementation in Task 3
```

Save to `src/core/inpaint.cpp`.

- [ ] **Step 4: Build and verify**

Run: `cmake --build build/mac-x64-Release 2>&1 | tail -5`
Expected: BUILD SUCCEEDED

- [ ] **Step 5: Commit**

```bash
git add src/detection/ncc_detector.hpp src/detection/ncc_detector.cpp src/core/inpaint.cpp CMakeLists.txt
git commit -m "feat: add NccDetector with three-stage detection (spatial, gradient, variance)"
```

---

### Task 3: Create Inpaint module

**Goal:** Implement the three traditional inpainting methods ported from GWT's `inpaint_residual`.

**Files:**
- Create: `src/core/inpaint.hpp`
- Modify: `src/core/inpaint.cpp` (replace stub)

**Acceptance Criteria:**
- [ ] Gaussian soft inpaint blends watermark boundary using gradient-weighted mask
- [ ] TELEA inpaint uses sparse gradient mask + cv::inpaint(INPAINT_TELEA)
- [ ] Navier-Stokes inpaint uses sparse gradient mask + cv::inpaint(INPAINT_NS)
- [ ] Strength parameter controls blend ratio between original and inpainted

**Verify:** `cmake --build build/mac-x64-Release 2>&1 | tail -5` → BUILD SUCCEEDED

**Steps:**

- [ ] **Step 1: Create `src/core/inpaint.hpp`**

```cpp
#pragma once

#include <opencv2/core.hpp>

namespace wmr {

enum class InpaintMethod {
    Gaussian,
    Telea,
    NavierStokes
};

struct InpaintConfig {
    float strength = 0.85f;
    InpaintMethod method = InpaintMethod::Gaussian;
    int radius = 10;
    int padding = 32;
};

void inpaint_residual(
    cv::Mat& image,
    const cv::Rect& region,
    const cv::Mat& alpha_map,
    const InpaintConfig& config = InpaintConfig{});

} // namespace wmr
```

- [ ] **Step 2: Replace `src/core/inpaint.cpp` stub with implementation**

```cpp
#include "core/inpaint.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>
#include <spdlog/spdlog.h>
#include <algorithm>

namespace wmr {

void inpaint_residual(
    cv::Mat& image,
    const cv::Rect& region,
    const cv::Mat& alpha_map,
    const InpaintConfig& config)
{
    if (image.empty() || region.width < 4 || region.height < 4) return;

    const float strength = std::clamp(config.strength, 0.0f, 1.0f);
    if (strength < 0.001f) return;

    // Padded region for context
    cv::Rect padded(
        region.x - config.padding,
        region.y - config.padding,
        region.width + config.padding * 2,
        region.height + config.padding * 2);
    padded &= cv::Rect(0, 0, image.cols, image.rows);

    if (padded.width < 8 || padded.height < 8) return;

    cv::Rect inner(
        region.x - padded.x,
        region.y - padded.y,
        region.width,
        region.height);
    inner &= cv::Rect(0, 0, padded.width, padded.height);

    // Compute alpha gradient for edge detection
    cv::Mat alpha_resized;
    int interp = (region.width > alpha_map.cols)
        ? cv::INTER_LINEAR : cv::INTER_AREA;
    cv::resize(alpha_map, alpha_resized,
               cv::Size(region.width, region.height), 0, 0, interp);

    cv::Mat grad_x, grad_y, grad_mag;
    cv::Sobel(alpha_resized, grad_x, CV_32F, 1, 0, 3);
    cv::Sobel(alpha_resized, grad_y, CV_32F, 0, 1, 3);
    cv::magnitude(grad_x, grad_y, grad_mag);

    double grad_min, grad_max;
    cv::minMaxLoc(grad_mag, &grad_min, &grad_max);
    if (grad_max <= grad_min) {
        spdlog::debug("inpaint: flat gradient, no edges found");
        return;
    }

    if (config.method == InpaintMethod::Gaussian) {
        // --- Gaussian Soft Inpaint ---
        cv::Mat grad_norm = (grad_mag - grad_min) / (grad_max - grad_min);
        cv::Mat grad_weight;
        cv::sqrt(grad_norm, grad_weight);

        // Dilate to cover residual spread
        cv::Mat dk = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
        cv::dilate(grad_weight, grad_weight, dk);
        cv::GaussianBlur(grad_weight, grad_weight, cv::Size(0, 0), 2.0);
        grad_weight *= strength;
        cv::threshold(grad_weight, grad_weight, 1.0, 1.0, cv::THRESH_TRUNC);

        // Embed into padded coordinate system
        cv::Mat weight = cv::Mat::zeros(padded.size(), CV_32F);
        grad_weight.copyTo(weight(inner));
        cv::GaussianBlur(weight, weight, cv::Size(0, 0), 1.0);

        // Gaussian blur the image
        int ksize = config.radius * 2 + 1;
        if (ksize % 2 == 0) ksize++;
        ksize = std::max(ksize, 3);
        double sigma = config.radius * 0.8;

        cv::Mat padded_area = image(padded).clone();
        cv::Mat blurred;
        cv::GaussianBlur(padded_area, blurred, cv::Size(ksize, ksize), sigma);

        // Per-pixel weighted blend
        cv::Mat dst = image(padded);
        cv::Mat weight_3ch;
        cv::merge(std::vector<cv::Mat>{weight, weight, weight}, weight_3ch);

        cv::Mat dst_f, blurred_f, result_f;
        dst.convertTo(dst_f, CV_32FC3);
        blurred.convertTo(blurred_f, CV_32FC3);

        cv::Mat one_minus_w = cv::Scalar(1.0, 1.0, 1.0) - weight_3ch;
        cv::multiply(dst_f, one_minus_w, dst_f);
        cv::multiply(blurred_f, weight_3ch, blurred_f);
        result_f = dst_f + blurred_f;

        result_f.convertTo(dst, CV_8UC3);

        int active = cv::countNonZero(weight > 0.01f);
        spdlog::debug("inpaint: Gaussian, strength={:.0f}%, {} active pixels",
                      strength * 100.0f, active);
    } else {
        // --- TELEA / Navier-Stokes ---
        cv::Mat grad_u8;
        grad_mag.convertTo(grad_u8, CV_8U,
                           255.0 / (grad_max - grad_min),
                           -grad_min * 255.0 / (grad_max - grad_min));

        cv::Mat sparse_mask;
        cv::threshold(grad_u8, sparse_mask, 20, 255, cv::THRESH_BINARY);

        cv::Mat dilate_kernel = cv::getStructuringElement(
            cv::MORPH_ELLIPSE, cv::Size(5, 5));
        cv::dilate(sparse_mask, sparse_mask, dilate_kernel);

        int masked = cv::countNonZero(sparse_mask);
        if (masked == 0) {
            spdlog::debug("inpaint: no edge pixels found, skipping");
            return;
        }

        // Embed mask into padded coordinate system
        cv::Mat mask = cv::Mat::zeros(padded.size(), CV_8UC1);
        sparse_mask.copyTo(mask(inner));

        cv::Mat padded_area = image(padded).clone();
        int cv_method = (config.method == InpaintMethod::Telea)
            ? cv::INPAINT_TELEA : cv::INPAINT_NS;

        cv::Mat inpainted;
        cv::inpaint(padded_area, mask, inpainted, config.radius, cv_method);

        // Blend at masked pixels only
        cv::Mat dst = image(padded);
        cv::Mat src_inner = dst(inner);
        cv::Mat inp_inner = inpainted(inner);
        cv::Mat mask_inner = mask(inner);

        if (strength >= 0.999f) {
            inp_inner.copyTo(src_inner, mask_inner);
        } else {
            cv::Mat blended;
            cv::addWeighted(src_inner, 1.0 - strength,
                            inp_inner, strength, 0.0, blended);
            blended.copyTo(src_inner, mask_inner);
        }

        const char* name = (config.method == InpaintMethod::Telea)
            ? "TELEA" : "Navier-Stokes";
        spdlog::debug("inpaint: {}, {} pixels repaired at {:.0f}% strength",
                      name, masked, strength * 100.0f);
    }
}

} // namespace wmr
```

- [ ] **Step 3: Build and verify**

Run: `cmake --build build/mac-x64-Release 2>&1 | tail -5`
Expected: BUILD SUCCEEDED

- [ ] **Step 4: Commit**

```bash
git add src/core/inpaint.hpp src/core/inpaint.cpp
git commit -m "feat: add inpaint module with Gaussian, TELEA, and Navier-Stokes methods"
```

---

### Task 4: Integrate detection + inpaint into WatermarkEngine

**Goal:** Add `detect_watermark` and `inpaint_residual` methods to WatermarkEngine, modify `remove_watermark` to optionally detect first and inpaint after.

**Files:**
- Modify: `src/core/watermark_engine.hpp`
- Modify: `src/core/watermark_engine.cpp`

**Acceptance Criteria:**
- [ ] `detect_watermark()` returns DetectionResult via NccDetector
- [ ] `remove_watermark()` has an overload that takes a detection result
- [ ] `inpaint_residual()` delegates to inpaint module
- [ ] Existing `remove_watermark(image, force_size)` still works (backward compat)

**Verify:** `cmake --build build/mac-x64-Release 2>&1 | tail -5` → BUILD SUCCEEDED

**Steps:**

- [ ] **Step 1: Update `src/core/watermark_engine.hpp`**

Replace entire file:

```cpp
#pragma once

#include <opencv2/core.hpp>
#include <optional>

#include "core/types.hpp"
#include "embedded_assets.hpp"

namespace wmr {

class NccDetector;
struct InpaintConfig;

class WatermarkEngine {
public:
    WatermarkEngine();

    void remove_watermark(cv::Mat& image,
                          std::optional<WatermarkSize> force_size = std::nullopt);

    void remove_watermark_detected(cv::Mat& image,
                                   const DetectionResult& detection);

    DetectionResult detect_watermark(
        const cv::Mat& image,
        std::optional<WatermarkSize> force_size = std::nullopt) const;

    void inpaint_residual(cv::Mat& image,
                          const cv::Rect& region,
                          const InpaintConfig& config = {}) const;

    void add_watermark(cv::Mat& image,
                       std::optional<WatermarkSize> force_size = std::nullopt);

    const cv::Mat& get_alpha_map(WatermarkSize size) const;

private:
    cv::Mat alpha_map_small_;
    cv::Mat alpha_map_large_;
    float logo_value_ = 255.0f;

    std::unique_ptr<NccDetector> detector_;

    cv::Mat create_interpolated_alpha(int width, int height, WatermarkSize size) const;
    void init_alpha_maps();
};

} // namespace wmr
```

- [ ] **Step 2: Update `src/core/watermark_engine.cpp`**

Replace entire file:

```cpp
#include "core/watermark_engine.hpp"
#include "core/blend_modes.hpp"
#include "core/inpaint.hpp"
#include "detection/ncc_detector.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>
#include <stdexcept>

namespace wmr {

WatermarkEngine::WatermarkEngine() {
    init_alpha_maps();
    detector_ = std::make_unique<NccDetector>(alpha_map_small_, alpha_map_large_);
}

void WatermarkEngine::init_alpha_maps() {
    std::vector<unsigned char> buf_small(
        embedded::bg_48_png, embedded::bg_48_png + embedded::bg_48_png_size);
    std::vector<unsigned char> buf_large(
        embedded::bg_96_png, embedded::bg_96_png + embedded::bg_96_png_size);

    cv::Mat bg_small = cv::imdecode(buf_small, cv::IMREAD_COLOR);
    if (bg_small.empty()) {
        throw std::runtime_error("Failed to decode embedded 48x48 background capture");
    }

    cv::Mat bg_large = cv::imdecode(buf_large, cv::IMREAD_COLOR);
    if (bg_large.empty()) {
        throw std::runtime_error("Failed to decode embedded 96x96 background capture");
    }

    if (bg_small.cols != 48 || bg_small.rows != 48) {
        cv::resize(bg_small, bg_small, cv::Size(48, 48), 0, 0, cv::INTER_AREA);
    }
    if (bg_large.cols != 96 || bg_large.rows != 96) {
        cv::resize(bg_large, bg_large, cv::Size(96, 96), 0, 0, cv::INTER_AREA);
    }

    alpha_map_small_ = calculate_alpha_map(bg_small);
    alpha_map_large_ = calculate_alpha_map(bg_large);

    spdlog::debug("Alpha maps initialized: small {}x{}, large {}x{}",
                  alpha_map_small_.cols, alpha_map_small_.rows,
                  alpha_map_large_.cols, alpha_map_large_.rows);
}

DetectionResult WatermarkEngine::detect_watermark(
    const cv::Mat& image,
    std::optional<WatermarkSize> force_size) const
{
    return detector_->detect(image, force_size);
}

void WatermarkEngine::remove_watermark(cv::Mat& image,
                                        std::optional<WatermarkSize> force_size) {
    if (image.empty()) {
        throw std::runtime_error("Empty image provided");
    }

    if (image.channels() == 4) {
        cv::cvtColor(image, image, cv::COLOR_BGRA2BGR);
    } else if (image.channels() == 1) {
        cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);
    }

    WatermarkSize size = force_size.value_or(
        get_watermark_size(image.cols, image.rows));

    auto config = get_watermark_config(image.cols, image.rows);
    if (force_size.has_value()) {
        if (*force_size == WatermarkSize::Small) {
            config = {32, 32, 48};
        } else {
            config = {64, 64, 96};
        }
    }

    cv::Point pos = config.get_position(image.cols, image.rows);
    const cv::Mat& alpha_map = get_alpha_map(size);

    spdlog::debug("Removing watermark at ({}, {}) size {}",
                  pos.x, pos.y, size == WatermarkSize::Small ? "48x48" : "96x96");

    remove_watermark_alpha_blend(image, alpha_map, pos, logo_value_);
}

void WatermarkEngine::remove_watermark_detected(
    cv::Mat& image,
    const DetectionResult& detection)
{
    if (image.empty()) return;

    if (image.channels() == 4) {
        cv::cvtColor(image, image, cv::COLOR_BGRA2BGR);
    } else if (image.channels() == 1) {
        cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);
    }

    const cv::Mat& alpha_map = get_alpha_map(detection.size);
    cv::Point pos(detection.region.x, detection.region.y);

    spdlog::debug("Removing detected watermark at ({}, {}) conf={:.2f}",
                  pos.x, pos.y, detection.confidence);

    remove_watermark_alpha_blend(image, alpha_map, pos, logo_value_);

    // Inpaint residuals
    inpaint_residual(image, detection.region);
}

void WatermarkEngine::inpaint_residual(
    cv::Mat& image,
    const cv::Rect& region,
    const InpaintConfig& config) const
{
    const cv::Mat& alpha_map = get_alpha_map(
        (region.width <= 48 && region.height <= 48)
            ? WatermarkSize::Small : WatermarkSize::Large);

    wmr::inpaint_residual(image, region, alpha_map, config);
}

void WatermarkEngine::add_watermark(cv::Mat& image,
                                     std::optional<WatermarkSize> force_size) {
    if (image.empty()) {
        throw std::runtime_error("Empty image provided");
    }

    if (image.channels() == 4) {
        cv::cvtColor(image, image, cv::COLOR_BGRA2BGR);
    } else if (image.channels() == 1) {
        cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);
    }

    WatermarkSize size = force_size.value_or(
        get_watermark_size(image.cols, image.rows));

    auto config = get_watermark_config(image.cols, image.rows);
    if (force_size.has_value()) {
        if (*force_size == WatermarkSize::Small) {
            config = {32, 32, 48};
        } else {
            config = {64, 64, 96};
        }
    }

    cv::Point pos = config.get_position(image.cols, image.rows);
    const cv::Mat& alpha_map = get_alpha_map(size);

    add_watermark_alpha_blend(image, alpha_map, pos, logo_value_);
}

const cv::Mat& WatermarkEngine::get_alpha_map(WatermarkSize size) const {
    return (size == WatermarkSize::Small) ? alpha_map_small_ : alpha_map_large_;
}

cv::Mat WatermarkEngine::create_interpolated_alpha(int width, int height,
                                                     WatermarkSize size) const {
    const cv::Mat& source = get_alpha_map(size);

    if (width == source.cols && height == source.rows) {
        return source.clone();
    }

    cv::Mat interpolated;
    int method = (width > source.cols || height > source.rows)
                     ? cv::INTER_LINEAR
                     : cv::INTER_AREA;

    cv::resize(source, interpolated, cv::Size(width, height), 0, 0, method);
    return interpolated;
}

} // namespace wmr
```

- [ ] **Step 3: Build and verify**

Run: `cmake --build build/mac-x64-Release 2>&1 | tail -5`
Expected: BUILD SUCCEEDED

- [ ] **Step 4: Commit**

```bash
git add src/core/watermark_engine.hpp src/core/watermark_engine.cpp
git commit -m "feat: integrate NCC detection and inpainting into WatermarkEngine"
```

---

### Task 5: Update CLI for detect→remove→inpaint pipeline

**Goal:** Wire up the new detection and inpainting features to the CLI. Add `--detect-only` flag. Default flow now: detect → remove → inpaint. `--force` skips detection.

**Files:**
- Modify: `src/cli/cli_app.hpp`
- Modify: `src/cli/cli_app.cpp`

**Acceptance Criteria:**
- [ ] `wmr input.png` detects watermark, removes if found, inpaints residuals
- [ ] `wmr input.png --force` skips detection (Phase 1 behavior)
- [ ] `wmr input.png --detect-only` reports detection result without modifying
- [ ] Non-watermarked images report detection failure and are not modified

**Verify:** `cmake --build build/mac-x64-Release 2>&1 | tail -5` → BUILD SUCCEEDED

**Steps:**

- [ ] **Step 1: Update `src/cli/cli_app.hpp`**

Replace entire file:

```cpp
#pragma once

#include <string>

namespace wmr {

struct CliOptions {
    std::string input_path;
    std::string output_path;
    bool force = false;
    bool force_small = false;
    bool force_large = false;
    bool verbose = false;
    bool detect_only = false;
    float inpaint_strength = 0.85f;
};

int run_cli(int argc, char* argv[]);

} // namespace wmr
```

- [ ] **Step 2: Update `src/cli/cli_app.cpp`**

Replace entire file:

```cpp
#include "cli/cli_app.hpp"
#include "core/watermark_engine.hpp"
#include "core/types.hpp"

#include <opencv2/imgcodecs.hpp>
#include <CLI/CLI.hpp>
#include <spdlog/spdlog.h>
#include <fmt/format.h>
#include <filesystem>

#ifndef APP_VERSION
#define APP_VERSION "0.1.0"
#endif

#ifndef APP_NAME
#define APP_NAME "wmr"
#endif

namespace wmr {

int run_cli(int argc, char* argv[]) {
    CLI::App app{"Watermark Remover — remove Gemini visible watermarks", APP_NAME};
    app.set_version_flag("-V,--version", APP_VERSION);

    CliOptions opts;

    app.add_option("input", opts.input_path, "Input image file")
        ->required()
        ->check(CLI::ExistingFile);

    app.add_option("-o,--output", opts.output_path, "Output file (default: overwrite in-place)");

    app.add_flag("-f,--force", opts.force, "Skip detection, force-remove at default position");
    app.add_flag("--force-small", opts.force_small, "Force 48x48 watermark");
    app.add_flag("--force-large", opts.force_large, "Force 96x96 watermark");
    app.add_flag("-v,--verbose", opts.verbose, "Verbose output");
    app.add_flag("--detect-only", opts.detect_only, "Report detection result without modifying");
    app.add_option("--inpaint-strength", opts.inpaint_strength,
                   "Inpaint strength 0.0-1.0 (default: 0.85)")
        ->check(CLI::Range(0.0f, 1.0f));

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError& e) {
        return app.exit(e);
    }

    if (opts.verbose) {
        spdlog::set_level(spdlog::level::debug);
    } else {
        spdlog::set_level(spdlog::level::info);
    }

    if (opts.force_small && opts.force_large) {
        spdlog::error("Cannot use both --force-small and --force-large");
        return 1;
    }

    try {
        WatermarkEngine engine;

        spdlog::info("Loading: {}", opts.input_path);
        cv::Mat image = cv::imread(opts.input_path, cv::IMREAD_COLOR);
        if (image.empty()) {
            spdlog::error("Failed to load image: {}", opts.input_path);
            return 1;
        }

        spdlog::info("Image: {}x{}", image.cols, image.rows);

        std::optional<WatermarkSize> force_size;
        if (opts.force_small) {
            force_size = WatermarkSize::Small;
        } else if (opts.force_large) {
            force_size = WatermarkSize::Large;
        }

        // --- Detect-only mode ---
        if (opts.detect_only) {
            auto result = engine.detect_watermark(image, force_size);
            if (result.detected) {
                spdlog::info("Watermark DETECTED (confidence: {:.1f}%)",
                             result.confidence * 100.0f);
                spdlog::info("  Region: ({}, {}) {}x{}",
                             result.region.x, result.region.y,
                             result.region.width, result.region.height);
                spdlog::info("  Scores: spatial={:.3f} grad={:.3f} var={:.3f}",
                             result.spatial_score, result.gradient_score,
                             result.variance_score);
            } else {
                spdlog::info("No watermark detected (confidence: {:.1f}%)",
                             result.confidence * 100.0f);
            }
            return result.detected ? 0 : 2;
        }

        // --- Normal processing: detect → remove → inpaint ---
        if (!opts.force) {
            auto detection = engine.detect_watermark(image, force_size);

            if (detection.detected) {
                spdlog::info("Watermark detected ({:.1f}% confidence), removing...",
                             detection.confidence * 100.0f);
                engine.remove_watermark_detected(image, detection);
            } else {
                spdlog::warn("No watermark detected ({:.1f}% confidence). "
                             "Use --force to remove anyway.",
                             detection.confidence * 100.0f);
                return 2;
            }
        } else {
            spdlog::info("Force mode: removing watermark at default position");
            engine.remove_watermark(image, force_size);
        }

        // Save output
        std::string output = opts.output_path.empty() ? opts.input_path : opts.output_path;
        std::filesystem::path out_path(output);
        if (!out_path.parent_path().empty() && !std::filesystem::exists(out_path.parent_path())) {
            std::filesystem::create_directories(out_path.parent_path());
        }

        std::vector<int> params;
        std::string ext = out_path.extension().string();
        if (ext == ".jpg" || ext == ".jpeg") {
            params = {cv::IMWRITE_JPEG_QUALITY, 100};
        } else if (ext == ".png") {
            params = {cv::IMWRITE_PNG_COMPRESSION, 6};
        } else if (ext == ".webp") {
            params = {cv::IMWRITE_WEBP_QUALITY, 101};
        }

        if (!cv::imwrite(output, image, params)) {
            spdlog::error("Failed to save: {}", output);
            return 1;
        }

        spdlog::info("Saved: {}", output);
        return 0;

    } catch (const std::exception& e) {
        spdlog::error("Error: {}", e.what());
        return 1;
    }
}

} // namespace wmr
```

- [ ] **Step 3: Build and verify**

Run: `cmake --build build/mac-x64-Release 2>&1 | tail -5`
Expected: BUILD SUCCEEDED

- [ ] **Step 4: Commit**

```bash
git add src/cli/cli_app.hpp src/cli/cli_app.cpp
git commit -m "feat: update CLI for detect→remove→inpaint pipeline with --detect-only flag"
```

---

### Task 6: Full verification and final commit

**Goal:** End-to-end testing of detection, removal, and inpainting against the test image.

**Files:** None (testing only)

**Acceptance Criteria:**
- [ ] `wmr --detect-only test-images/2400x1792-gemini.png` reports DETECTED with confidence > 0.7
- [ ] `wmr test-images/2400x1792-gemini.png -o test-output-phase2.png` produces a clean image
- [ ] `wmr --force test-images/2400x1792-gemini.png -o test-output-force.png` still works (Phase 1 compat)
- [ ] Detection on a non-watermarked image reports not detected

**Verify:** All commands above succeed with expected output.

**Steps:**

- [ ] **Step 1: Test detection on watermarked image**

Run: `./build/mac-x64-Release/wmr --detect-only -v test-images/2400x1792-gemini.png`
Expected: "Watermark DETECTED" with confidence > 70%

- [ ] **Step 2: Test full detect→remove→inpaint pipeline**

Run: `./build/mac-x64-Release/wmr -v test-images/2400x1792-gemini.png -o test-output-phase2.png`
Expected: "Saved: test-output-phase2.png", watermark removed with inpainted boundary

- [ ] **Step 3: Test --force backward compatibility**

Run: `./build/mac-x64-Release/wmr --force test-images/2400x1792-gemini.png -o test-output-force.png`
Expected: "Saved: test-output-force.png", watermark removed (same as Phase 1 behavior)

- [ ] **Step 4: Verify help output**

Run: `./build/mac-x64-Release/wmr --help`
Expected: Shows `--detect-only`, `--force`, `--inpaint-strength` options

- [ ] **Step 5: Final commit if any fixes were needed**

```bash
git add -A
git commit -m "fix: Phase 2 verification adjustments"
```

---

## Self-Review

### Spec Coverage

| Spec Requirement | Task |
|-----------------|------|
| Three-stage NCC (spatial, gradient, variance) | Task 2 |
| Circuit breaker at spatial < 0.25 | Task 2 |
| Heuristic fusion (0.50/0.30/0.20 weights) | Task 2 |
| DetectionResult struct | Task 1 |
| Gaussian soft inpaint | Task 3 |
| TELEA inpaint | Task 3 |
| Navier-Stokes inpaint | Task 3 |
| Sequential application (Gaussian first) | Task 4 (inpaint_residual called after remove) |
| CLI `--detect-only` flag | Task 5 |
| CLI `--force` skips detection | Task 5 |
| Fallback if confidence < threshold | Task 5 (warns and exits with code 2) |

### Placeholder Scan

No TBDs, TODOs, or incomplete sections found.

### Type Consistency

- `DetectionResult` defined in Task 1 (types.hpp), used in Tasks 2, 4, 5 — fields match
- `InpaintConfig` defined in Task 3 (inpaint.hpp), used in Task 4 — fields match
- `NccDetector::detect()` returns `DetectionResult`, called by `WatermarkEngine::detect_watermark()` — consistent
- `WatermarkEngine::remove_watermark_detected()` takes `const DetectionResult&` — matches Task 5 CLI usage
