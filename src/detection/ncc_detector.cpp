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
