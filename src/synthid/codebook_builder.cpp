#include "synthid/codebook_builder.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>
#include <filesystem>
#include <vector>
#include <cmath>

namespace fs = std::filesystem;

namespace wmr {

CodebookBuilder::CodebookBuilder(FftContext& fft)
    : fft_(fft) {}

BuildStats CodebookBuilder::build_from_directory(
    const std::string& dir_path,
    const std::string& output_path)
{
    BuildStats stats;

    if (!fs::exists(dir_path) || !fs::is_directory(dir_path)) {
        throw std::runtime_error("Not a valid directory: " + dir_path);
    }

    std::map<std::pair<int,int>, ProfileAccumulator> accums;

    for (const auto& entry : fs::directory_iterator(dir_path)) {
        if (!entry.is_regular_file()) continue;

        auto ext = entry.path().extension().string();
        for (auto& c : ext) c = static_cast<char>(std::tolower(c));

        if (ext != ".png" && ext != ".jpg" && ext != ".jpeg" && ext != ".webp") continue;

        cv::Mat img = cv::imread(entry.path().string(), cv::IMREAD_COLOR);
        if (img.empty()) {
            spdlog::warn("Skipping unreadable: {}", entry.path().filename().string());
            continue;
        }

        accumulate(img, accums);
        ++stats.total_images;
        spdlog::debug("Accumulated: {} ({}x{})",
                      entry.path().filename().string(), img.cols, img.rows);
    }

    if (stats.total_images == 0) {
        throw std::runtime_error("No images found in: " + dir_path);
    }

    SpectralCodebook codebook;

    for (auto& [key, acc] : accums) {
        if (acc.count < 3) {
            spdlog::warn("Resolution {}x{} has only {} samples (minimum 3 recommended)",
                         acc.width, acc.height, acc.count);
            ++stats.skipped_low_samples;
        }

        SpectralProfile profile = finalize(acc);
        codebook.add_profile(profile);
        ++stats.profiles_created;

        spdlog::info("Profile: {}x{} ({} samples)", acc.width, acc.height, acc.count);
    }

    codebook.save(output_path);
    spdlog::info("Saved codebook: {} profiles → {}", stats.profiles_created, output_path);

    return stats;
}

void CodebookBuilder::accumulate(
    const cv::Mat& image,
    std::map<std::pair<int,int>, ProfileAccumulator>& accums) const
{
    cv::Mat work;
    if (image.channels() == 4) {
        cv::cvtColor(image, work, cv::COLOR_BGRA2BGR);
    } else if (image.channels() == 1) {
        cv::cvtColor(image, work, cv::COLOR_GRAY2BGR);
    } else {
        work = image.clone();
    }

    int w = work.cols;
    int h = work.rows;
    auto key = std::make_pair(h, w);

    auto& acc = accums[key];
    acc.width = w;
    acc.height = h;

    cv::Mat channels[3];
    cv::split(work, channels);

    for (int ch = 0; ch < 3; ++ch) {
        cv::Mat ch_float;
        channels[ch].convertTo(ch_float, CV_32FC1, 1.0 / 255.0);

        cv::Mat ch_fft = fft_.forward(ch_float);
        cv::Mat mag = FftContext::magnitude(ch_fft);
        cv::Mat ph = FftContext::phase(ch_fft);

        if (acc.count == 0) {
            acc.mag_sum[ch] = cv::Mat::zeros(mag.size(), CV_32FC1);
            acc.phase_sum[ch] = cv::Mat::zeros(ph.size(), CV_32FC1);
            acc.mag_sq_sum[ch] = cv::Mat::zeros(mag.size(), CV_32FC1);
        }

        acc.mag_sum[ch] += mag;
        acc.mag_sq_sum[ch] += mag.mul(mag);
        acc.phase_sum[ch] += ph;
    }

    ++acc.count;
}

SpectralProfile CodebookBuilder::finalize(const ProfileAccumulator& acc) const {
    SpectralProfile profile;
    profile.width = acc.width;
    profile.height = acc.height;
    profile.sample_count = acc.count;

    float n = static_cast<float>(acc.count);

    for (int ch = 0; ch < 3; ++ch) {
        // Average magnitude
        profile.magnitude_bgr[ch] = acc.mag_sum[ch] / n;

        // Average phase
        profile.phase_bgr[ch] = acc.phase_sum[ch] / n;

        // Consistency: standard deviation of magnitude
        // std = sqrt(E[X^2] - (E[X])^2)
        cv::Mat mean_sq = profile.magnitude_bgr[ch].mul(profile.magnitude_bgr[ch]);
        cv::Mat variance = acc.mag_sq_sum[ch] / n - mean_sq;
        cv::sqrt(cv::max(variance, 0.0), profile.consistency_bgr[ch]);
    }

    return profile;
}

} // namespace wmr
