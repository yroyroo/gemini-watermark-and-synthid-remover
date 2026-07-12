#pragma once

#include <opencv2/core.hpp>
#include <string>

namespace wmr {

// Per-scene gate logic for NotebookLM adaptive inpaint dispatch. Pure logic
// (operates on cv::Mat, no VideoReader/FFmpeg) so it can be unit-tested without
// linking FFmpeg (which the test target excludes).

// Background complexity score in a band AROUND the mark (the mark bbox itself
// is excluded, since the watermark strokes would inflate the gradient energy).
// Higher = more intricate/textured background. Based on mean Sobel gradient
// magnitude in the surrounding ring.
float background_complexity_score(const cv::Mat& gray_frame, const cv::Rect& mark_rect);

// Convenience: true when the background around the mark is intricate enough to
// warrant a stronger inpainter than plain NS.
bool background_is_intricate(const cv::Mat& gray_frame, const cv::Rect& mark_rect,
                             float threshold);

// Resolve the per-scene inpaint method from the scene's background complexity.
// Pure (no VideoReader) so it can be unit-tested without linking FFmpeg.
//   complexity: scene background-complexity score
//   threshold:  intricate when complexity >= threshold (--complexity-threshold,
//               default 15)
//   has_migan:  whether the build compiled the MI-GAN inpainter (WMR_AI_MIGAN);
//               passed as a bool so the function is testable in either config.
// Returns "migan" when intricate AND MI-GAN is compiled in — the default
// intricate-scene method — otherwise "ns". There is NO user method choice: the
// pipeline always uses NS (uniform) + MI-GAN (intricate).
std::string resolve_inpaint_method(float complexity, double threshold, bool has_migan);

} // namespace wmr
