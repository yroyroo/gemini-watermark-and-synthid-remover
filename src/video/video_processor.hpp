#pragma once

#include <string>
#include <cstdint>
#include <chrono>
#include <opencv2/core.hpp>

#include "core/types.hpp"
#include "video/video_writer.hpp"

namespace wmr {

enum class VideoWatermarkProfile {
    GeminiDiamond,
    VeoLegacy,
};

struct VideoWatermarkConfig {
    VideoWatermarkProfile profile = VideoWatermarkProfile::GeminiDiamond;
    std::string variant;
    bool force = false;
    float inpaint_strength = 0.85f;
};

struct VideoResult {
    int64_t frames_processed = 0;
    int64_t frames_skipped = 0;
    float detection_confidence = 0.0f;
    double elapsed_seconds = 0.0;
    bool success = false;
    std::string message;
};

class VideoProcessor {
public:
    VideoResult process(const std::string& input_path,
                        const std::string& output_path,
                        const VideoWatermarkConfig& config,
                        const EncodeOptions& encode_opts = {});

private:
    static constexpr int kShotSampleCount = 12;
    static constexpr double kShotCoverage = 0.9;       // sample first 90% of video
    static constexpr int kPositionTolerancePx = 4;      // ±4px from shot anchor
    static constexpr float kOcclusionGateNcc = 0.35f;   // skip frames below this NCC
    static constexpr int kProgressIntervalFrames = 100;

    struct ShotDetection {
        bool found = false;
        cv::Point position;
        WatermarkSize size = WatermarkSize::Small;
        float confidence = 0.0f;
        cv::Rect region;
    };

    ShotDetection detect_in_shot(class VideoReader& reader,
                                 class WatermarkEngine& engine,
                                 const VideoWatermarkConfig& config);
};

} // namespace wmr
