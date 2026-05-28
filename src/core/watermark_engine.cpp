#include "core/watermark_engine.hpp"
#include "core/blend_modes.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>
#include <stdexcept>

namespace wmr {

WatermarkEngine::WatermarkEngine() {
    init_alpha_maps();
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
    // Override config if forcing a specific size
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
