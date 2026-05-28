#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "core/fft_context.hpp"
#include <opencv2/core.hpp>

using namespace wmr;
using Catch::Matchers::WithinAbs;

TEST_CASE("FFT forward produces CV_32FC2 output", "[fft]") {
    FftContext fft;
    cv::Mat input(64, 64, CV_32FC1, cv::Scalar(0.5f));
    cv::Mat output = fft.forward(input);

    REQUIRE(output.type() == CV_32FC2);
    REQUIRE(output.rows == 64);
    REQUIRE(output.cols == 64);
}

TEST_CASE("FFT round-trip preserves signal", "[fft]") {
    FftContext fft;
    cv::Mat input(32, 32, CV_32FC1);
    for (int y = 0; y < 32; ++y) {
        for (int x = 0; x < 32; ++x) {
            input.at<float>(y, x) = 0.3f + 0.2f * std::sin(y * 0.5f) * std::cos(x * 0.5f);
        }
    }

    cv::Mat freq = fft.forward(input);
    cv::Mat recovered = fft.inverse(freq);

    REQUIRE(recovered.type() == CV_32FC1);
    REQUIRE(recovered.size() == input.size());

    // Check round-trip error
    cv::Mat diff;
    cv::absdiff(input, recovered, diff);
    double max_err = 0;
    cv::minMaxLoc(diff, nullptr, &max_err);
    REQUIRE(max_err < 1e-5);
}

TEST_CASE("FFT magnitude and phase extraction", "[fft]") {
    FftContext fft;
    cv::Mat input(16, 16, CV_32FC1, cv::Scalar(1.0f));
    cv::Mat freq = fft.forward(input);

    cv::Mat mag = FftContext::magnitude(freq);
    cv::Mat ph = FftContext::phase(freq);

    REQUIRE(mag.type() == CV_32FC1);
    REQUIRE(ph.type() == CV_32FC1);
    REQUIRE(mag.size() == freq.size());

    // DC component should be dominant for constant input
    float dc_mag = mag.at<float>(0, 0);
    REQUIRE(dc_mag > 100.0f);  // 16*16 = 256
}

TEST_CASE("FFT round-trip at 1024x1024 (production size)", "[fft]") {
    FftContext fft;
    cv::Mat input(1024, 1024, CV_32FC1);
    for (int y = 0; y < 1024; ++y) {
        for (int x = 0; x < 1024; ++x) {
            input.at<float>(y, x) = 0.3f + 0.2f * std::sin(y * 0.1f) * std::cos(x * 0.1f);
        }
    }

    cv::Mat freq = fft.forward(input);
    cv::Mat recovered = fft.inverse(freq);

    REQUIRE(recovered.type() == CV_32FC1);
    REQUIRE(recovered.size() == input.size());

    cv::Mat diff;
    cv::absdiff(input, recovered, diff);
    double max_err = 0;
    cv::minMaxLoc(diff, nullptr, &max_err);
    REQUIRE(max_err < 1e-4);
}

TEST_CASE("from_polar reconstructs complex from magnitude and phase", "[fft]") {
    FftContext fft;
    cv::Mat input(16, 16, CV_32FC1);
    for (int y = 0; y < 16; ++y) {
        for (int x = 0; x < 16; ++x) {
            input.at<float>(y, x) = static_cast<float>(y * 16 + x) / 256.0f;
        }
    }

    cv::Mat freq = fft.forward(input);
    cv::Mat mag = FftContext::magnitude(freq);
    cv::Mat ph = FftContext::phase(freq);

    cv::Mat reconstructed = FftContext::from_polar(mag, ph);

    // Check reconstruction matches original complex
    REQUIRE(reconstructed.type() == CV_32FC2);
    REQUIRE(reconstructed.size() == freq.size());

    // Compare real and imaginary parts
    cv::Mat channels[2], orig_channels[2];
    cv::split(reconstructed, channels);
    cv::split(freq, orig_channels);

    cv::Mat diff_real, diff_imag;
    cv::absdiff(channels[0], orig_channels[0], diff_real);
    cv::absdiff(channels[1], orig_channels[1], diff_imag);

    double max_real_err = 0, max_imag_err = 0;
    cv::minMaxLoc(diff_real, nullptr, &max_real_err);
    cv::minMaxLoc(diff_imag, nullptr, &max_imag_err);

    REQUIRE(max_real_err < 1e-2);
    REQUIRE(max_imag_err < 1e-2);
}
