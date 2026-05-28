#include "core/blend_modes.hpp"

#include <opencv2/imgproc.hpp>
#include <algorithm>

namespace wmr {

cv::Mat calculate_alpha_map(const cv::Mat& bg_capture) {
    CV_Assert(!bg_capture.empty());

    cv::Mat gray;
    if (bg_capture.channels() >= 3) {
        std::vector<cv::Mat> channels;
        cv::split(bg_capture, channels);
        cv::max(channels[0], channels[1], gray);
        cv::max(gray, channels[2], gray);
    } else {
        gray = bg_capture.clone();
    }

    cv::Mat alpha_map;
    gray.convertTo(alpha_map, CV_32FC1, 1.0 / 255.0);
    return alpha_map;
}

void remove_watermark_alpha_blend(
    cv::Mat& image,
    const cv::Mat& alpha_map,
    const cv::Point& position,
    float logo_value)
{
    CV_Assert(!image.empty() && !alpha_map.empty());
    CV_Assert(image.channels() == 3);
    CV_Assert(alpha_map.type() == CV_32FC1);

    int x = position.x;
    int y = position.y;
    int w = alpha_map.cols;
    int h = alpha_map.rows;

    int x1 = std::max(0, x);
    int y1 = std::max(0, y);
    int x2 = std::min(image.cols, x + w);
    int y2 = std::min(image.rows, y + h);

    if (x1 >= x2 || y1 >= y2) return;

    cv::Rect alpha_roi(x1 - x, y1 - y, x2 - x1, y2 - y1);
    cv::Rect image_roi(x1, y1, x2 - x1, y2 - y1);

    cv::Mat alpha_region = alpha_map(alpha_roi);
    cv::Mat image_region = image(image_roi);

    cv::Mat image_f;
    image_region.convertTo(image_f, CV_32FC3);

    constexpr float alpha_threshold = 0.002f;
    constexpr float max_alpha = 0.99f;

    for (int row = 0; row < image_f.rows; ++row) {
        const float* alpha_ptr = alpha_region.ptr<float>(row);
        cv::Vec3f* img_ptr = image_f.ptr<cv::Vec3f>(row);

        for (int col = 0; col < image_f.cols; ++col) {
            float alpha = alpha_ptr[col];
            if (alpha < alpha_threshold) continue;

            alpha = std::min(alpha, max_alpha);
            float one_minus_alpha = 1.0f - alpha;

            for (int c = 0; c < 3; ++c) {
                float watermarked = img_ptr[col][c];
                float original = (watermarked - alpha * logo_value) / one_minus_alpha;
                img_ptr[col][c] = std::clamp(original, 0.0f, 255.0f);
            }
        }
    }

    cv::Mat result;
    image_f.convertTo(result, CV_8UC3);
    result.copyTo(image_region);
}

void add_watermark_alpha_blend(
    cv::Mat& image,
    const cv::Mat& alpha_map,
    const cv::Point& position,
    float logo_value)
{
    CV_Assert(!image.empty() && !alpha_map.empty());
    CV_Assert(image.channels() == 3);
    CV_Assert(alpha_map.type() == CV_32FC1);

    int x = position.x;
    int y = position.y;
    int w = alpha_map.cols;
    int h = alpha_map.rows;

    int x1 = std::max(0, x);
    int y1 = std::max(0, y);
    int x2 = std::min(image.cols, x + w);
    int y2 = std::min(image.rows, y + h);

    if (x1 >= x2 || y1 >= y2) return;

    cv::Rect alpha_roi(x1 - x, y1 - y, x2 - x1, y2 - y1);
    cv::Rect image_roi(x1, y1, x2 - x1, y2 - y1);

    cv::Mat alpha_region = alpha_map(alpha_roi);
    cv::Mat image_region = image(image_roi);

    cv::Mat image_f;
    image_region.convertTo(image_f, CV_32FC3);

    constexpr float alpha_threshold = 0.002f;

    for (int row = 0; row < image_f.rows; ++row) {
        const float* alpha_ptr = alpha_region.ptr<float>(row);
        cv::Vec3f* img_ptr = image_f.ptr<cv::Vec3f>(row);

        for (int col = 0; col < image_f.cols; ++col) {
            float alpha = alpha_ptr[col];
            if (alpha < alpha_threshold) continue;

            float one_minus_alpha = 1.0f - alpha;

            for (int c = 0; c < 3; ++c) {
                float original = img_ptr[col][c];
                float blended = alpha * logo_value + one_minus_alpha * original;
                img_ptr[col][c] = std::clamp(blended, 0.0f, 255.0f);
            }
        }
    }

    cv::Mat result;
    image_f.convertTo(result, CV_8UC3);
    result.copyTo(image_region);
}

} // namespace wmr
