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
