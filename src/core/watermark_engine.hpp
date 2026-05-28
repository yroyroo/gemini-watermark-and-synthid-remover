#pragma once

#include <opencv2/core.hpp>
#include <optional>

#include "core/types.hpp"
#include "embedded_assets.hpp"

namespace wmr {

class WatermarkEngine {
public:
    WatermarkEngine();

    void remove_watermark(cv::Mat& image,
                          std::optional<WatermarkSize> force_size = std::nullopt);
    void add_watermark(cv::Mat& image,
                       std::optional<WatermarkSize> force_size = std::nullopt);

    const cv::Mat& get_alpha_map(WatermarkSize size) const;

private:
    cv::Mat alpha_map_small_;
    cv::Mat alpha_map_large_;
    float logo_value_ = 255.0f;

    cv::Mat create_interpolated_alpha(int width, int height, WatermarkSize size) const;
    void init_alpha_maps();
};

} // namespace wmr
