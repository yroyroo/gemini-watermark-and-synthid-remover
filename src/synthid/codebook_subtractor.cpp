#include "synthid/codebook_subtractor.hpp"

#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cmath>

namespace wmr {

CodebookSubtractor::CodebookSubtractor(FftContext& fft)
    : fft_(fft) {}

auto CodebookSubtractor::get_strength_params(RemovalStrength strength) -> StrengthParams {
    switch (strength) {
        case RemovalStrength::Gentle:
            return {0.60f, 0.70f, 0.50f, 30.0f};
        case RemovalStrength::Moderate:
            return {0.80f, 0.50f, 0.70f, 25.0f};
        case RemovalStrength::Aggressive:
            return {0.95f, 0.30f, 0.90f, 20.0f};
        case RemovalStrength::Maximum:
            return {1.00f, 0.15f, 0.95f, 15.0f};
    }
    return {0.80f, 0.50f, 0.70f, 25.0f};
}

void CodebookSubtractor::remove_synthid(
    cv::Mat& image,
    const SpectralCodebook& codebook,
    const RemovalConfig& config)
{
    if (image.empty()) return;

    if (image.channels() == 4) {
        cv::cvtColor(image, image, cv::COLOR_BGRA2BGR);
    } else if (image.channels() == 1) {
        cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);
    }

    const int h = image.rows;
    const int w = image.cols;

    const auto& profile = codebook.get_profile(w, h);

    // Determine multi-pass schedule
    RemovalStrength base_strength = config.strength;
    if (config.custom_strength >= 0.0f) {
        if (config.custom_strength <= 0.25f) base_strength = RemovalStrength::Gentle;
        else if (config.custom_strength <= 0.50f) base_strength = RemovalStrength::Moderate;
        else if (config.custom_strength <= 0.75f) base_strength = RemovalStrength::Aggressive;
        else base_strength = RemovalStrength::Maximum;
    }

    // Multi-pass schedule: aggressive→moderate→gentle
    struct PassSchedule {
        RemovalStrength level;
        int count;
    };

    PassSchedule schedule[4];
    int num_passes = 0;

    switch (base_strength) {
        case RemovalStrength::Gentle:
            schedule[0] = {RemovalStrength::Gentle, 1};
            num_passes = 1;
            break;
        case RemovalStrength::Moderate:
            schedule[0] = {RemovalStrength::Moderate, 1};
            schedule[1] = {RemovalStrength::Gentle, 1};
            num_passes = 2;
            break;
        case RemovalStrength::Aggressive:
            schedule[0] = {RemovalStrength::Aggressive, 1};
            schedule[1] = {RemovalStrength::Moderate, 1};
            schedule[2] = {RemovalStrength::Gentle, 1};
            num_passes = 3;
            break;
        case RemovalStrength::Maximum:
            schedule[0] = {RemovalStrength::Maximum, 1};
            schedule[1] = {RemovalStrength::Aggressive, 1};
            schedule[2] = {RemovalStrength::Moderate, 1};
            num_passes = 3;
            break;
    }

    // Convert to float
    cv::Mat work;
    image.convertTo(work, CV_32FC3, 1.0 / 255.0);

    // Compute luminance for adaptive blending
    cv::Scalar mean_val = cv::mean(work);
    float luminance = static_cast<float>((mean_val[0] + mean_val[1] + mean_val[2]) / 3.0);

    bool needs_resize = (profile.width != w || profile.height != h);

    // Per-channel processing
    cv::Mat channels[3];
    cv::split(work, channels);

    spdlog::info("SynthID removal: {}x{} image, {} passes, base_strength={}",
                 w, h, num_passes, static_cast<int>(base_strength));

    for (int pass = 0; pass < num_passes; ++pass) {
        auto params = get_strength_params(schedule[pass].level);

        spdlog::debug("Pass {}/{}: removal={:.2f}, cons_floor={:.2f}, "
                      "mag_cap={:.2f}, dc_radius={:.0f}",
                      pass + 1, num_passes, params.removal, params.cons_floor,
                      params.mag_cap, params.dc_radius);

        for (int ch = 0; ch < 3; ++ch) {
            // Resize profile to match image if needed
            cv::Mat prof_mag, prof_phase, prof_cons;
            if (needs_resize) {
                cv::resize(profile.magnitude_bgr[ch], prof_mag, {w, h}, 0, 0, cv::INTER_LINEAR);
                cv::resize(profile.phase_bgr[ch], prof_phase, {w, h}, 0, 0, cv::INTER_LINEAR);
                cv::resize(profile.consistency_bgr[ch], prof_cons, {w, h}, 0, 0, cv::INTER_LINEAR);
            } else {
                prof_mag = profile.magnitude_bgr[ch];
                prof_phase = profile.phase_bgr[ch];
                prof_cons = profile.consistency_bgr[ch];
            }

            cv::Mat img_fft = fft_.forward(channels[ch]);

            SpectralProfile ch_profile;
            ch_profile.width = w;
            ch_profile.height = h;
            ch_profile.magnitude_bgr[0] = prof_mag;
            ch_profile.phase_bgr[0] = prof_phase;
            ch_profile.consistency_bgr[0] = prof_cons;

            cv::Mat wm_estimate = estimate_watermark_fft(
                img_fft, ch, params.removal, params.cons_floor,
                params.mag_cap, params.dc_radius,
                ch_profile, luminance);

            cv::Mat cleaned_fft;
            cv::subtract(img_fft, wm_estimate, cleaned_fft);

            channels[ch] = fft_.inverse(cleaned_fft);
        }
    }

    // Anti-aliasing
    cv::Mat merged;
    cv::merge(channels, 3, merged);
    cv::GaussianBlur(merged, merged, {3, 3}, 0.4);

    // Clamp and convert back
    merged = cv::max(merged, 0.0);
    merged = cv::min(merged, 1.0);
    merged.convertTo(image, CV_8UC3, 255.0);

    spdlog::debug("SynthID removal complete: {} passes", num_passes);
}

cv::Mat CodebookSubtractor::estimate_watermark_fft(
    const cv::Mat& image_fft,
    int channel,
    float removal_factor,
    float cons_floor,
    float mag_cap,
    float dc_radius,
    const SpectralProfile& profile,
    float /*image_luminance*/)
{
    const int rows = image_fft.rows;
    const int cols = image_fft.cols;
    const float ch_weight = kChannelWeights[channel];

    cv::Mat prof_mag = profile.magnitude_bgr[0];
    cv::Mat prof_phase = profile.phase_bgr[0];
    cv::Mat prof_cons = profile.consistency_bgr[0];

    // DC exclusion ramp
    cv::Mat dc_ramp(rows, cols, CV_32FC1);
    for (int y = 0; y < rows; ++y) {
        float fy = static_cast<float>(y);
        if (fy > rows / 2.0f) fy -= rows;
        for (int x = 0; x < cols; ++x) {
            float fx = static_cast<float>(x);
            if (fx > cols / 2.0f) fx -= cols;
            float dist = std::sqrt(fy * fy + fx * fx);
            dc_ramp.at<float>(y, x) = std::clamp(dist / dc_radius, 0.0f, 1.0f);
        }
    }

    // Consistency weighting
    cv::Mat cons_weight;
    cv::subtract(prof_cons, cons_floor, cons_weight);
    cv::divide(cons_weight, (1.0f - cons_floor + 1e-9f), cons_weight);
    cv::max(cons_weight, 0.0, cons_weight);
    cv::min(cons_weight, 1.0, cons_weight);

    // Subtract magnitude
    cv::Mat subtract_mag = prof_mag.mul(cons_weight) * removal_factor * ch_weight;

    // Apply DC ramp
    subtract_mag = subtract_mag.mul(dc_ramp);

    // Safety cap: never subtract more than mag_cap * |image_fft|
    cv::Mat img_mag = FftContext::magnitude(image_fft);
    cv::Mat cap;
    img_mag.copyTo(cap);
    cap *= mag_cap;
    cv::min(subtract_mag, cap, subtract_mag);

    // Construct complex watermark estimate using profile phase
    cv::Mat wm_estimate = FftContext::from_polar(subtract_mag, prof_phase);

    return wm_estimate;
}

} // namespace wmr
