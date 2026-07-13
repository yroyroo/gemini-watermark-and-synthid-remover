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

// Resolve the per-scene inpaint method. Pure (no VideoReader) so it can be
// unit-tested without linking FFmpeg — deliberately arch-agnostic (the platform
// default is passed IN, not read from a build macro, so the test target — which
// never defines WMR_AI_MIGAN/WMR_AI_MIGAN_COREML — sees the same logic).
//   complexity:        scene background-complexity score (ignored when not gating)
//   threshold:         intricate when complexity >= threshold (--complexity-threshold, default 15)
//   has_migan:         whether the build compiled the MI-GAN inpainter (WMR_AI_MIGAN);
//                      passed as a bool so the function is testable in either config.
//   requested:         "auto" (defer to platform_default) | "ns" | "migan"
//                      (--notebooklm-method; "auto" is the CLI default).
//   platform_default:  "migan" -> MI-GAN everywhere, ignore complexity (Apple Silicon,
//                      where the ANE makes MI-GAN fast); "gated" -> NS for uniform,
//                      MI-GAN for intricate (the 1.9.0 behavior, used elsewhere).
// Returns "migan" or "ns". An explicit "ns"/"migan" request wins, subject to
// has_migan (forced "migan" with has_migan=false falls back to "ns"). "auto" +
// platform_default="migan" returns "migan" regardless of complexity. NS is the
// universal fallback (also covers a runtime model-load failure, which the caller
// catches separately via MiganInpainter::is_ready()).
std::string resolve_inpaint_method(float complexity, double threshold, bool has_migan,
                                   const std::string& requested,
                                   const std::string& platform_default);

} // namespace wmr
