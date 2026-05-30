#include "video/video_processor.hpp"
#include "video/video_reader.hpp"
#include "core/watermark_engine.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

#include <spdlog/spdlog.h>
#include <fmt/format.h>

namespace wmr {

// ---------------------------------------------------------------------------
// Shot-level detection: sample frames and establish watermark baseline
// ---------------------------------------------------------------------------
VideoProcessor::ShotDetection VideoProcessor::detect_in_shot(
    VideoReader& reader,
    WatermarkEngine& engine,
    const VideoWatermarkConfig& /*config*/)
{
    ShotDetection result;
    const int64_t total = reader.frame_count();
    if (total <= 0) {
        spdlog::warn("Video has no frames, shot detection skipped");
        return result;
    }

    // Determine sample indices: evenly spaced over first 90% of the video
    const int64_t coverage_end = static_cast<int64_t>(
        static_cast<double>(total) * kShotCoverage);
    const int sample_count = static_cast<int>(
        std::min(static_cast<int64_t>(kShotSampleCount), coverage_end));

    if (sample_count <= 0) {
        spdlog::warn("Video too short for shot sampling");
        return result;
    }

    // For a single sample, just check frame 0
    if (sample_count == 1) {
        cv::Mat frame;
        if (reader.seek(0) && reader.next_frame(frame) && !frame.empty()) {
            auto det = engine.detect_watermark(frame);
            if (det.detected) {
                result.found = true;
                result.position = cv::Point(det.region.x, det.region.y);
                result.size = det.size;
                result.confidence = det.confidence;
                result.region = det.region;
                spdlog::info("Shot detection (single frame): detected at ({},{}) conf={:.2f}",
                             result.position.x, result.position.y, result.confidence);
            } else {
                auto wconfig = get_watermark_config(reader.width(), reader.height());
                result.position = wconfig.get_position(reader.width(), reader.height());
                result.size = get_watermark_size(reader.width(), reader.height());
            }
        }
        return result;
    }

    struct Sample {
        cv::Point position;
        WatermarkSize size;
        float confidence;
        cv::Rect region;
    };

    std::vector<Sample> detections;
    detections.reserve(sample_count);

    spdlog::info("Shot detection: sampling {} frames over first {} of {} total",
                 sample_count, coverage_end, total);

    for (int i = 0; i < sample_count; ++i) {
        int64_t frame_idx = static_cast<int64_t>(
            static_cast<double>(i) / static_cast<double>(sample_count - 1) *
            static_cast<double>(coverage_end - 1));

        if (!reader.seek(frame_idx)) {
            spdlog::debug("Shot sample {}: seek to {} failed", i, frame_idx);
            continue;
        }

        cv::Mat frame;
        if (!reader.next_frame(frame) || frame.empty()) {
            spdlog::debug("Shot sample {}: read at {} failed", i, frame_idx);
            continue;
        }

        auto det = engine.detect_watermark(frame);
        if (det.detected) {
            detections.push_back({cv::Point(det.region.x, det.region.y),
                                  det.size, det.confidence, det.region});
            spdlog::debug("Shot sample {}: detected at ({},{}) conf={:.2f}",
                          i, det.region.x, det.region.y, det.confidence);
        } else {
            spdlog::debug("Shot sample {}: not detected ({:.2f})",
                          i, det.confidence);
        }
    }

    // Need >50% detection rate to trust the result
    const int threshold = (sample_count + 1) / 2;  // majority
    if (static_cast<int>(detections.size()) < threshold) {
        spdlog::info("Shot detection: {}/{} samples detected (< {} majority threshold)",
                     detections.size(), sample_count, threshold);
        // Fall back to default position
        auto wconfig = get_watermark_config(reader.width(), reader.height());
        result.position = wconfig.get_position(reader.width(), reader.height());
        result.size = get_watermark_size(reader.width(), reader.height());
        result.found = false;
        result.confidence = 0.0f;
        return result;
    }

    // Compute median of detected positions, sizes, and confidences
    std::vector<int> xs, ys;
    std::vector<float> confs;
    int small_count = 0;
    cv::Rect median_region;

    for (const auto& d : detections) {
        xs.push_back(d.position.x);
        ys.push_back(d.position.y);
        confs.push_back(d.confidence);
        if (d.size == WatermarkSize::Small) ++small_count;
    }

    const auto mid = xs.size() / 2;

    std::sort(xs.begin(), xs.end());
    std::sort(ys.begin(), ys.end());
    std::sort(confs.begin(), confs.end());

    result.position = cv::Point(xs[mid], ys[mid]);
    result.confidence = confs[mid];

    // Use the region from the detection closest to median position
    int best_dist = std::numeric_limits<int>::max();
    for (const auto& d : detections) {
        int dist = std::abs(d.position.x - result.position.x) +
                   std::abs(d.position.y - result.position.y);
        if (dist < best_dist) {
            best_dist = dist;
            median_region = d.region;
        }
    }
    result.region = median_region;

    // Majority size
    result.size = (small_count > static_cast<int>(detections.size()) / 2)
                      ? WatermarkSize::Small : WatermarkSize::Large;
    result.found = true;

    spdlog::info("Shot detection: anchor at ({},{}) size={} conf={:.2f} "
                 "({}/{} detections)",
                 result.position.x, result.position.y,
                 result.size == WatermarkSize::Small ? "small" : "large",
                 result.confidence, detections.size(), sample_count);

    return result;
}

// ---------------------------------------------------------------------------
// Main processing loop
// ---------------------------------------------------------------------------
VideoResult VideoProcessor::process(const std::string& input_path,
                                    const std::string& output_path,
                                    const VideoWatermarkConfig& config,
                                    const EncodeOptions& encode_opts)
{
    VideoResult result;
    const auto t_start = std::chrono::steady_clock::now();

    // Open input
    VideoReader reader;
    if (!reader.open(input_path)) {
        result.success = false;
        result.message = fmt::format("Failed to open input: {}", input_path);
        spdlog::error("{}", result.message);
        return result;
    }

    spdlog::info("Input: {} ({}x{}, {:.2f} fps, {} frames)",
                 input_path, reader.width(), reader.height(),
                 reader.fps(), reader.frame_count());

    WatermarkEngine engine;

    // Shot-level detection
    ShotDetection shot;
    if (config.force) {
        auto wconfig = get_watermark_config(reader.width(), reader.height());
        shot.found = false;
        shot.position = wconfig.get_position(reader.width(), reader.height());
        shot.size = get_watermark_size(reader.width(), reader.height());
        shot.confidence = 1.0f;
        shot.region = cv::Rect(shot.position.x, shot.position.y,
                               wconfig.logo_size, wconfig.logo_size);
        spdlog::info("Force mode: using default position ({},{})",
                     shot.position.x, shot.position.y);
    } else {
        shot = detect_in_shot(reader, engine, config);
        reader.seek(0);
    }

    // Open output writer (audio streams set up before MP4 header)
    VideoWriter writer;
    if (!writer.open(output_path, reader.width(), reader.height(),
                     reader.fps(), encode_opts, input_path)) {
        result.success = false;
        result.message = fmt::format("Failed to open output: {}", output_path);
        spdlog::error("{}", result.message);
        reader.close();
        return result;
    }

    // Process frames
    cv::Mat frame;
    int64_t frame_idx = 0;
    int64_t last_progress_frame = 0;
    auto t_last_progress = t_start;

    while (reader.next_frame(frame)) {
        if (frame.empty()) {
            spdlog::warn("Frame {}: empty, skipping", frame_idx);
            writer.write_frame(frame);
            ++result.frames_skipped;
            ++frame_idx;
            continue;
        }

        // Guard against corrupt frames from seek artifacts
        if (frame.cols != reader.width() || frame.rows != reader.height()) {
            spdlog::warn("Frame {}: unexpected dimensions {}x{} (expected {}x{}), skipping",
                         frame_idx, frame.cols, frame.rows,
                         reader.width(), reader.height());
            ++result.frames_skipped;
            ++frame_idx;
            continue;
        }

        if (config.force) {
            // Force mode: always remove at anchor position
            DetectionResult det;
            det.detected = true;
            det.confidence = 1.0f;
            det.region = shot.region;
            det.size = shot.size;
            engine.remove_watermark_detected(frame, det);
            writer.write_frame(frame);
            ++result.frames_processed;
        } else {
            // Detect watermark in current frame (NCC refinement)
            auto detection = engine.detect_watermark(frame);

            if (!detection.detected ||
                detection.confidence < kOcclusionGateNcc) {
                // Occlusion gate: watermark not reliably detected
                spdlog::debug("Frame {}: occluded/skipped (conf={:.2f})",
                              frame_idx, detection.confidence);
                writer.write_frame(frame);
                ++result.frames_skipped;
            } else {
                // Position refinement: check against shot anchor
                cv::Point detected_pos(detection.region.x, detection.region.y);
                int dx = std::abs(detected_pos.x - shot.position.x);
                int dy = std::abs(detected_pos.y - shot.position.y);

                if (dx > kPositionTolerancePx || dy > kPositionTolerancePx) {
                    // Too far from anchor — use anchor position instead
                    spdlog::debug("Frame {}: detection ({},{}) too far from anchor "
                                  "({},{}) — using anchor",
                                  frame_idx, detected_pos.x, detected_pos.y,
                                  shot.position.x, shot.position.y);
                    detection.region.x = shot.position.x;
                    detection.region.y = shot.position.y;
                    detection.size = shot.size;
                }

                engine.remove_watermark_detected(frame, detection);
                writer.write_frame(frame);
                ++result.frames_processed;
            }
        }

        // Progress output
        ++frame_idx;
        auto t_now = std::chrono::steady_clock::now();
        double since_last = std::chrono::duration<double>(t_now - t_last_progress).count();

        if (frame_idx - last_progress_frame >= kProgressIntervalFrames ||
            since_last >= 2.0) {
            double elapsed = std::chrono::duration<double>(t_now - t_start).count();
            double proc_fps = static_cast<double>(frame_idx) / std::max(elapsed, 1e-9);
            int64_t remaining = reader.frame_count() - frame_idx;
            double eta = (remaining > 0) ? (static_cast<double>(remaining) / proc_fps) : 0.0;

            spdlog::info("Frame {}/{} ({:.1f} fps, ETA {:.0f}s)",
                         frame_idx, reader.frame_count(), proc_fps, eta);

            last_progress_frame = frame_idx;
            t_last_progress = t_now;
        }
    }

    // Copy audio
    if (!writer.copy_audio()) {
        spdlog::warn("Audio copy failed or no audio stream present");
    }

    writer.close();
    reader.close();

    // Build result
    const auto t_end = std::chrono::steady_clock::now();
    result.elapsed_seconds = std::chrono::duration<double>(t_end - t_start).count();
    result.detection_confidence = shot.confidence;
    result.success = true;
    result.message = fmt::format("Processed {} frames ({} skipped) in {:.1f}s",
                                 result.frames_processed, result.frames_skipped,
                                 result.elapsed_seconds);

    spdlog::info("Done: {} processed, {} skipped, detection conf={:.2f}, "
                 "elapsed={:.1f}s",
                 result.frames_processed, result.frames_skipped,
                 result.detection_confidence, result.elapsed_seconds);

    return result;
}

} // namespace wmr
