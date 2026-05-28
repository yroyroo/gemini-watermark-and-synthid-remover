#include "core/fft_context.hpp"

#include <fftw3.h>
#include <stdexcept>

namespace wmr {

FftContext::FftContext() = default;

FftContext::~FftContext() {
    for (auto& [key, plan] : plans_) {
        if (plan) {
            fftwf_destroy_plan(reinterpret_cast<fftwf_plan>(plan));
        }
    }
    plans_.clear();
}

cv::Mat FftContext::forward(const cv::Mat& channel) {
    if (channel.type() != CV_32FC1) {
        throw std::runtime_error("FftContext::forward requires CV_32FC1 input");
    }

    const int rows = channel.rows;
    const int cols = channel.cols;

    // Create complex input (real part = channel data, imag part = 0)
    cv::Mat complex_in(rows, cols, CV_32FC2);
    cv::Mat channels_in[2] = {channel.clone(), cv::Mat::zeros(rows, cols, CV_32FC1)};
    cv::merge(channels_in, 2, complex_in);

    cv::Mat output(rows, cols, CV_32FC2);

    PlanKey key{rows, cols, true};
    auto it = plans_.find(key);
    fftwf_plan plan;
    if (it != plans_.end()) {
        plan = reinterpret_cast<fftwf_plan>(it->second);
    } else {
        plan = fftwf_plan_dft_2d(rows, cols,
                                  reinterpret_cast<fftwf_complex*>(complex_in.ptr<float>()),
                                  reinterpret_cast<fftwf_complex*>(output.ptr<float>()),
                                  FFTW_FORWARD, FFTW_MEASURE);
        if (!plan) {
            throw std::runtime_error("FFTW forward plan creation failed");
        }
        plans_[key] = reinterpret_cast<void*>(plan);
    }

    fftwf_execute_dft(plan,
                       reinterpret_cast<fftwf_complex*>(complex_in.ptr<float>()),
                       reinterpret_cast<fftwf_complex*>(output.ptr<float>()));

    return output;
}

cv::Mat FftContext::inverse(const cv::Mat& complex) {
    if (complex.type() != CV_32FC2) {
        throw std::runtime_error("FftContext::inverse requires CV_32FC2 input");
    }

    const int rows = complex.rows;
    const int cols = complex.cols;

    cv::Mat output = complex.clone();

    PlanKey key{rows, cols, false};
    auto it = plans_.find(key);
    fftwf_plan plan;
    if (it != plans_.end()) {
        plan = reinterpret_cast<fftwf_plan>(it->second);
    } else {
        plan = fftwf_plan_dft_2d(rows, cols,
                                  reinterpret_cast<fftwf_complex*>(output.ptr<float>()),
                                  reinterpret_cast<fftwf_complex*>(output.ptr<float>()),
                                  FFTW_BACKWARD, FFTW_MEASURE);
        if (!plan) {
            throw std::runtime_error("FFTW inverse plan creation failed");
        }
        plans_[key] = reinterpret_cast<void*>(plan);
    }

    fftwf_execute_dft(plan,
                       reinterpret_cast<fftwf_complex*>(output.ptr<float>()),
                       reinterpret_cast<fftwf_complex*>(output.ptr<float>()));

    // Normalize and extract real part
    output /= static_cast<float>(rows * cols);

    cv::Mat channels_out[2];
    cv::split(output, channels_out);
    return channels_out[0];  // real part only
}

cv::Mat FftContext::magnitude(const cv::Mat& complex) {
    cv::Mat channels[2];
    cv::split(complex, channels);
    cv::Mat mag;
    cv::magnitude(channels[0], channels[1], mag);
    return mag;
}

cv::Mat FftContext::phase(const cv::Mat& complex) {
    cv::Mat channels[2];
    cv::split(complex, channels);
    cv::Mat ph;
    cv::phase(channels[0], channels[1], ph);
    return ph;
}

cv::Mat FftContext::from_polar(const cv::Mat& mag, const cv::Mat& ph) {
    cv::Mat real, imag;
    cv::polarToCart(mag, ph, real, imag);
    cv::Mat result;
    cv::merge(std::vector<cv::Mat>{real, imag}, result);
    return result;
}

} // namespace wmr
