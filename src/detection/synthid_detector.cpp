#include "detection/synthid_detector.hpp"

#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cmath>

namespace wmr {

SynthidDetector::SynthidDetector(FftContext& fft)
    : fft_(fft) {}

SynthidDetectionResult SynthidDetector::detect(
    const cv::Mat& image,
    const SpectralCodebook& codebook) const
{
    SynthidDetectionResult result;

    if (image.empty()) return result;

    cv::Mat work;
    if (image.channels() == 4) {
        cv::cvtColor(image, work, cv::COLOR_BGRA2BGR);
    } else if (image.channels() == 1) {
        cv::cvtColor(image, work, cv::COLOR_GRAY2BGR);
    } else {
        work = image.clone();
    }

    const int w = work.cols;
    const int h = work.rows;

    const SpectralProfile& profile = codebook.get_profile(w, h);

    // Convert to float for FFT operations
    cv::Mat work_f;
    work.convertTo(work_f, CV_32FC3, 1.0 / 255.0);

    cv::Mat channels[3];
    cv::split(work_f, channels);

    // Per-channel scores, averaged across BGR
    float avg_noise = 0.0f;
    float avg_phase = 0.0f;
    float avg_struct = 0.0f;

    for (int ch = 0; ch < 3; ++ch) {
        cv::Mat prof_mag, prof_phase;
        if (profile.width != w || profile.height != h) {
            cv::resize(profile.magnitude_bgr[ch], prof_mag, {w, h}, 0, 0, cv::INTER_LINEAR);
            cv::resize(profile.phase_bgr[ch], prof_phase, {w, h}, 0, 0, cv::INTER_LINEAR);
        } else {
            prof_mag = profile.magnitude_bgr[ch];
            prof_phase = profile.phase_bgr[ch];
        }

        avg_noise += noise_correlation(channels[ch], prof_mag);

        cv::Mat ch_fft = fft_.forward(channels[ch]);
        avg_phase += carrier_phase_matching(ch_fft, prof_phase);
        avg_struct += structure_ratio(ch_fft, prof_mag);
    }

    avg_noise /= 3.0f;
    avg_phase /= 3.0f;
    avg_struct /= 3.0f;

    float ms_cons = multi_scale_consistency(work, profile);

    result.noise_correlation = avg_noise;
    result.carrier_phase_score = avg_phase;
    result.structure_ratio = avg_struct;
    result.multi_scale_consistency = ms_cons;

    result.confidence = kWeightNoiseCorr * avg_noise
                      + kWeightCarrierPhase * avg_phase
                      + kWeightStructure * avg_struct
                      + kWeightMultiScale * ms_cons;

    result.confidence = std::clamp(result.confidence, 0.0f, 1.0f);
    result.detected = result.confidence >= kDefaultThreshold;

    spdlog::debug("SynthID detect: noise={:.3f} phase={:.3f} struct={:.3f} "
                  "ms_cons={:.3f} → confidence={:.3f} ({})",
                  avg_noise, avg_phase, avg_struct, ms_cons,
                  result.confidence,
                  result.detected ? "DETECTED" : "not detected");

    return result;
}

float SynthidDetector::noise_correlation(
    const cv::Mat& channel,
    const cv::Mat& profile_mag) const
{
    // Extract noise residual via bilateral filter denoising
    cv::Mat denoised;
    cv::bilateralFilter(channel, denoised, 9, 75.0, 75.0);

    cv::Mat noise = channel - denoised;

    // FFT of noise residual
    cv::Mat noise_fft = fft_.forward(noise);
    cv::Mat noise_mag = FftContext::magnitude(noise_fft);

    // Normalized cross-correlation between noise spectrum and profile magnitude
    cv::Mat a, b;
    noise_mag.convertTo(a, CV_32FC1);
    profile_mag.convertTo(b, CV_32FC1);

    if (a.size() != b.size()) {
        cv::resize(a, a, b.size());
    }

    cv::Scalar mean_a = cv::mean(a);
    cv::Scalar mean_b = cv::mean(b);

    a -= mean_a;
    b -= mean_b;

    double dot = a.dot(b);
    double norm_a = std::sqrt(a.dot(a));
    double norm_b = std::sqrt(b.dot(b));

    if (norm_a < 1e-9 || norm_b < 1e-9) return 0.0f;

    // NCC in range [-1, 1] → map to [0, 1]
    float ncc = static_cast<float>(dot / (norm_a * norm_b));
    return std::clamp((ncc + 1.0f) * 0.5f, 0.0f, 1.0f);
}

float SynthidDetector::carrier_phase_matching(
    const cv::Mat& channel_fft,
    const cv::Mat& profile_phase) const
{
    cv::Mat img_phase = FftContext::phase(channel_fft);

    cv::Mat a, b;
    img_phase.convertTo(a, CV_32FC1);
    profile_phase.convertTo(b, CV_32FC1);

    if (a.size() != b.size()) {
        cv::resize(a, a, b.size());
    }

    // Phase coherence: cosine of phase difference
    cv::Mat phase_diff;
    cv::subtract(a, b, phase_diff);
    cv::Mat cos_vals(phase_diff.size(), CV_32FC1);
    for (int y = 0; y < phase_diff.rows; ++y) {
        const float* src = phase_diff.ptr<float>(y);
        float* dst = cos_vals.ptr<float>(y);
        for (int x = 0; x < phase_diff.cols; ++x) {
            dst[x] = std::cos(src[x]);
        }
    }

    // Average cosine similarity → 1.0 means perfect match
    float coherence = static_cast<float>(cv::mean(cos_vals)[0]);

    // Map from [-1, 1] to [0, 1]
    return std::clamp((coherence + 1.0f) * 0.5f, 0.0f, 1.0f);
}

float SynthidDetector::structure_ratio(
    const cv::Mat& channel_fft,
    const cv::Mat& profile_mag) const
{
    cv::Mat img_mag = FftContext::magnitude(channel_fft);

    cv::Mat a, b;
    img_mag.convertTo(a, CV_32FC1);
    profile_mag.convertTo(b, CV_32FC1);

    if (a.size() != b.size()) {
        cv::resize(a, a, b.size());
    }

    // Energy where profile is strong (carrier bins) vs total energy
    cv::Mat carrier_mask;
    cv::threshold(b, carrier_mask, 0.01, 1.0, cv::THRESH_BINARY);

    cv::Mat carrier_energy, total_energy;
    cv::multiply(a, carrier_mask, carrier_energy);
    double carrier_sum = cv::sum(carrier_energy)[0];
    double total_sum = cv::sum(a)[0];

    if (total_sum < 1e-9) return 0.0f;

    float ratio = static_cast<float>(carrier_sum / total_sum);

    // SynthID typically puts ~2-5% of spectral energy in carrier bins
    // Normalize: ratio > 0.03 is strong signal
    return std::clamp(ratio / 0.06f, 0.0f, 1.0f);
}

float SynthidDetector::multi_scale_consistency(
    const cv::Mat& image,
    const SpectralProfile& /*profile*/) const
{
    // Run carrier phase detection at multiple scales
    // SynthID is resolution-dependent → consistent score across scales = real signal

    cv::Mat channels[3];
    cv::split(image, channels);

    auto phase_score_at_scale = [&](float scale) -> float {
        cv::Mat scaled;
        cv::resize(image, scaled, {}, scale, scale, cv::INTER_LINEAR);

        cv::Mat ch[3];
        cv::split(scaled, ch);

        float total_phase = 0.0f;
        for (int c = 0; c < 3; ++c) {
            cv::Mat ch_float;
            ch[c].convertTo(ch_float, CV_32FC1, 1.0 / 255.0);
            cv::Mat ch_fft = fft_.forward(ch_float);
            cv::Mat ph = FftContext::phase(ch_fft);

            // Measure phase concentration (low std = coherent = watermark signal)
            cv::Scalar mean_ph, std_ph;
            cv::meanStdDev(ph, mean_ph, std_ph);
            // Lower std → more coherent → higher score
            float coherence = 1.0f - static_cast<float>(std_ph[0] / CV_PI);
            total_phase += std::clamp(coherence, 0.0f, 1.0f);
        }
        return total_phase / 3.0f;
    };

    float score_1x = phase_score_at_scale(1.0f);
    float score_half = phase_score_at_scale(0.5f);
    float score_quarter = phase_score_at_scale(0.25f);

    // Consistency: how similar are the scores across scales
    float mean_score = (score_1x + score_half + score_quarter) / 3.0f;
    float variance = ((score_1x - mean_score) * (score_1x - mean_score)
                    + (score_half - mean_score) * (score_half - mean_score)
                    + (score_quarter - mean_score) * (score_quarter - mean_score)) / 3.0f;

    // Low variance (consistent) AND decent mean → strong signal
    float consistency = 1.0f - std::min(std::sqrt(variance) * 3.0f, 1.0f);
    return std::clamp(mean_score * consistency * 2.0f, 0.0f, 1.0f);
}

} // namespace wmr
