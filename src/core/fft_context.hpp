#pragma once

#include <opencv2/core.hpp>
#include <map>
#include <utility>

namespace wmr {

class FftContext {
public:
    FftContext();
    ~FftContext();

    // Forward 2D FFT: CV_32FC1 -> CV_32FC2 (complex)
    cv::Mat forward(const cv::Mat& channel);

    // Inverse 2D FFT: CV_32FC2 (complex) -> CV_32FC1
    cv::Mat inverse(const cv::Mat& complex);

    // Get magnitude from complex: CV_32FC2 -> CV_32FC1
    static cv::Mat magnitude(const cv::Mat& complex);

    // Get phase from complex: CV_32FC2 -> CV_32FC1
    static cv::Mat phase(const cv::Mat& complex);

    // Construct complex from magnitude and phase: CV_32FC1, CV_32FC1 -> CV_32FC2
    static cv::Mat from_polar(const cv::Mat& mag, const cv::Mat& ph);

private:
    struct PlanKey {
        int rows, cols;
        bool forward;
        bool operator<(const PlanKey& o) const {
            return std::tie(rows, cols, forward) < std::tie(o.rows, o.cols, o.forward);
        }
    };

    std::map<PlanKey, void*> plans_;  // fftw_plan handles
};

} // namespace wmr
