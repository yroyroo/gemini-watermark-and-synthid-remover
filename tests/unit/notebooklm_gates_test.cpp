#include <catch2/catch_test_macros.hpp>

#include "video/notebooklm_gates.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

using namespace wmr;

static cv::Rect corner_mark() { return {200, 200, 60, 12}; }

TEST_CASE("NotebookLM complexity: uniform background scores ~0", "[notebooklm]") {
    cv::Mat uniform(240, 320, CV_8UC1, cv::Scalar(110));
    float s = background_complexity_score(uniform, corner_mark());
    REQUIRE(s < 1.0f);  // no gradients -> ~0
}

TEST_CASE("NotebookLM complexity: textured background scores high", "[notebooklm]") {
    cv::Mat tex(240, 320, CV_8UC1);
    cv::RNG rng(12345);
    rng.fill(tex, cv::RNG::UNIFORM, 0, 256);  // random noise -> high gradient energy
    cv::Mat uniform(240, 320, CV_8UC1, cv::Scalar(110));
    float s_textured = background_complexity_score(tex, corner_mark());
    float s_uniform = background_complexity_score(uniform, corner_mark());
    REQUIRE(s_textured > s_uniform + 20.0f);  // clearly higher
}

TEST_CASE("NotebookLM complexity: mark bbox edge excluded from the band", "[notebooklm]") {
    // Uniform background, but a high-contrast mark drawn INSIDE the bbox. The
    // band has a gap from the mark, so the mark's edge must NOT inflate the score.
    cv::Mat img(240, 320, CV_8UC1, cv::Scalar(110));
    cv::Rect mark = corner_mark();
    cv::rectangle(img, mark, cv::Scalar(255), cv::FILLED);
    float s = background_complexity_score(img, mark);
    REQUIRE(s < 1.0f);
}

TEST_CASE("NotebookLM complexity: intricate classification", "[notebooklm]") {
    cv::Rect mark = corner_mark();
    cv::Mat uniform(240, 320, CV_8UC1, cv::Scalar(110));
    REQUIRE_FALSE(background_is_intricate(uniform, mark, 10.0f));

    cv::Mat tex(240, 320, CV_8UC1);
    cv::RNG rng(777);
    rng.fill(tex, cv::RNG::UNIFORM, 0, 256);
    REQUIRE(background_is_intricate(tex, mark, 10.0f));
}

TEST_CASE("NotebookLM complexity: accepts BGR input", "[notebooklm]") {
    cv::Mat uniform_bgr(240, 320, CV_8UC3, cv::Scalar(110, 120, 130));
    float s = background_complexity_score(uniform_bgr, corner_mark());
    REQUIRE(s < 1.0f);
}

TEST_CASE("NotebookLM method routing: resolve_inpaint_method", "[notebooklm]") {
    const double thr = 15.0;
    const float uniform = 5.0f;     // below threshold (cinematic-like)
    const float intricate = 41.0f;  // above threshold (Neon-like explainer)

    SECTION("auto + uniform -> ns") {
        REQUIRE(resolve_inpaint_method(uniform, thr, "auto", true) == "ns");
        REQUIRE(resolve_inpaint_method(uniform, thr, "auto", false) == "ns");
    }
    SECTION("auto + intricate + xphoto -> fsr") {
        REQUIRE(resolve_inpaint_method(intricate, thr, "auto", true) == "fsr");
    }
    SECTION("auto + intricate + NO xphoto -> ns (no regression vs v1.6.0 NS default)") {
        REQUIRE(resolve_inpaint_method(intricate, thr, "auto", false) == "ns");
    }
    SECTION("explicit ns -> ns regardless of complexity/xphoto") {
        REQUIRE(resolve_inpaint_method(intricate, thr, "ns", true) == "ns");
        REQUIRE(resolve_inpaint_method(uniform, thr, "ns", false) == "ns");
    }
    SECTION("explicit fsr honoured only when xphoto is compiled in") {
        REQUIRE(resolve_inpaint_method(intricate, thr, "fsr", true) == "fsr");
        REQUIRE(resolve_inpaint_method(uniform, thr, "fsr", true) == "fsr");
        REQUIRE(resolve_inpaint_method(intricate, thr, "fsr", false) == "ns");
    }
    SECTION("complexity == threshold counts as intricate (>=)") {
        REQUIRE(resolve_inpaint_method(static_cast<float>(thr), thr, "auto", true) == "fsr");
    }
    SECTION("unrecognized requested value behaves like auto (safe)") {
        REQUIRE(resolve_inpaint_method(intricate, thr, "bogus", true) == "fsr");
        REQUIRE(resolve_inpaint_method(uniform, thr, "bogus", true) == "ns");
    }
}
