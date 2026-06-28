#ifdef WMR_AI_DENOISE

#include <catch2/catch_test_macros.hpp>

#include "core/ai_denoise.hpp"
#include "core/blend_modes.hpp"
#include "core/watermark_engine.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <cmath>

using namespace wmr;

namespace {

// Thin wrapper so REQUIRE_NOTHROW has a single callable to invoke.
void denoise_impl_smoke(
    NcnnDenoiser& denoiser,
    cv::Mat& image,
    const cv::Rect& region,
    const cv::Mat& alpha_map)
{
    denoiser.denoise(image, region, alpha_map, /*sigma=*/25.0f, /*strength=*/0.85f, /*padding=*/16);
}

} // namespace

// ============================================================================
// Model load / runtime init
// ============================================================================

TEST_CASE("NcnnDenoiser loads the embedded FDnCNN model", "[aidenoise]") {
    NcnnDenoiser denoiser;

    REQUIRE(denoiser.initialize());
    REQUIRE(denoiser.is_ready());

    // The runtime must report either a GPU device name or a CPU thread count.
    const std::string device = denoiser.device_name();
    INFO("device_name(): " << device);
    REQUIRE(!device.empty());

    // Log which path was taken so CI runs surface CPU-vs-GPU in the output.
    INFO("is_gpu_enabled(): " << (denoiser.is_gpu_enabled() ? "GPU" : "CPU"));
}

// ============================================================================
// Denoise smoke test
// ============================================================================

TEST_CASE("NcnnDenoiser runs inference on a forward-blended ROI", "[aidenoise]") {
    NcnnDenoiser denoiser;
    REQUIRE(denoiser.initialize());
    REQUIRE(denoiser.is_ready());

    // Synthetic image: smooth gradient so the model has real content to denoise.
    cv::Mat image(96, 96, CV_8UC3);
    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            const cv::Vec3b px(
                static_cast<uchar>((x * 255) / image.cols),
                static_cast<uchar>((y * 255) / image.rows),
                static_cast<uchar>(((x + y) * 255) / (image.cols + image.rows)));
            image.at<cv::Vec3b>(y, x) = px;
        }
    }

    // Known alpha map (CV_32FC1, [0,1]) over a 48x48 watermark region.
    const cv::Rect region(40, 40, 48, 48);
    cv::Mat alpha_map(region.size(), CV_32FC1);
    for (int y = 0; y < alpha_map.rows; ++y) {
        for (int x = 0; x < alpha_map.cols; ++x) {
            // Soft disk: high in the centre, falling off at the edges — this
            // produces a non-trivial gradient the denoiser's edge mask can lock
            // onto, rather than a flat alpha which has zero gradient.
            const float dx = (x - alpha_map.cols * 0.5f) / (alpha_map.cols * 0.5f);
            const float dy = (y - alpha_map.rows * 0.5f) / (alpha_map.rows * 0.5f);
            const float r = std::sqrt(dx * dx + dy * dy);
            alpha_map.at<float>(y, x) = std::max(0.0f, 0.8f * (1.0f - r));
        }
    }

    // Forward-blend a watermark onto the region (this is what reverse-blend later
    // leaves residuals on), so the denoise call has a realistic input.
    add_watermark_alpha_blend(image, alpha_map, region.tl(), 255.0f);

    cv::Mat blended = image.clone();

    // Must run to completion without throwing or crashing.
    REQUIRE_NOTHROW(denoise_impl_smoke(denoiser, blended, region, alpha_map));

    // The call modified the image in-place; just assert it remains a valid
    // BGR CV_8UC3 of the same geometry. We deliberately do not over-assert
    // pixel values — the smoke test only proves the path is wired up.
    REQUIRE(blended.type() == CV_8UC3);
    REQUIRE(blended.rows == image.rows);
    REQUIRE(blended.cols == image.cols);
}

// ============================================================================
// Engine wiring: AI denoise is the default cleanup path when built
// ============================================================================
TEST_CASE("Engine remove_watermark_detected dispatches to AI cleanup", "[aidenoise]") {
    cv::Mat original(512, 512, CV_8UC3);
    for (int y = 0; y < original.rows; ++y)
        for (int x = 0; x < original.cols; ++x)
            original.at<cv::Vec3b>(y, x) = cv::Vec3b(
                static_cast<uchar>((x * 255) / original.cols),
                static_cast<uchar>((y * 255) / original.rows),
                128);

    WatermarkEngine engine;
    const cv::Mat& alpha = engine.get_alpha_map(WatermarkSize::Large);  // V1 96x96
    const cv::Point pos(original.cols - 64 - alpha.cols, original.rows - 64 - alpha.rows);
    cv::Mat watermarked = original.clone();
    add_watermark_alpha_blend(watermarked, alpha, pos, 255.0f);

    // Build a detection at the known region — we're testing the cleanup dispatch
    // (not the detector), so bypass detection heuristics.
    DetectionResult det;
    det.detected = true;
    det.confidence = 0.9f;
    det.region = cv::Rect(pos.x, pos.y, alpha.cols, alpha.rows);
    det.size = WatermarkSize::Large;

    // Default InpaintConfig.method == AiDenoise when built -> exercises the
    // engine's AI dispatch (with Gaussian fallback if init fails). Must not throw.
    cv::Mat out = watermarked.clone();
    REQUIRE_NOTHROW(engine.remove_watermark_detected(out, det, 0.85f, &alpha));

    REQUIRE(out.type() == CV_8UC3);
    REQUIRE(out.rows == watermarked.rows);
    REQUIRE(out.cols == watermarked.cols);

    // The watermark region was processed (reverse-blend altered it).
    cv::Rect roi(pos.x, pos.y, alpha.cols, alpha.rows);
    cv::Mat diff;
    cv::absdiff(watermarked(roi), out(roi), diff);
    REQUIRE(cv::mean(diff)[0] > 0.0f);
}

#endif // WMR_AI_DENOISE
