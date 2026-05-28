#pragma once

#include <opencv2/core.hpp>

namespace wmr {

cv::Mat calculate_alpha_map(const cv::Mat& bg_capture);

void remove_watermark_alpha_blend(
    cv::Mat& image,
    const cv::Mat& alpha_map,
    const cv::Point& position,
    float logo_value = 255.0f);

void add_watermark_alpha_blend(
    cv::Mat& image,
    const cv::Mat& alpha_map,
    const cv::Point& position,
    float logo_value = 255.0f);

} // namespace wmr
