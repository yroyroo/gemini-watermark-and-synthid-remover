#include "cli/cli_app.hpp"
#include "core/watermark_engine.hpp"
#include "core/types.hpp"

#include <opencv2/imgcodecs.hpp>
#include <CLI/CLI.hpp>
#include <spdlog/spdlog.h>
#include <fmt/format.h>
#include <filesystem>

#ifndef APP_VERSION
#define APP_VERSION "0.1.0"
#endif

#ifndef APP_NAME
#define APP_NAME "wmr"
#endif

namespace wmr {

int run_cli(int argc, char* argv[]) {
    CLI::App app{"Watermark Remover — remove Gemini visible watermarks", APP_NAME};
    app.set_version_flag("-V,--version", APP_VERSION);

    CliOptions opts;

    app.add_option("input", opts.input_path, "Input image file")
        ->required()
        ->check(CLI::ExistingFile);

    app.add_option("-o,--output", opts.output_path, "Output file (default: overwrite in-place)");

    app.add_flag("-f,--force", opts.force, "Process unconditionally (skip detection)");
    app.add_flag("--force-small", opts.force_small, "Force 48x48 watermark");
    app.add_flag("--force-large", opts.force_large, "Force 96x96 watermark");
    app.add_flag("-v,--verbose", opts.verbose, "Verbose output");

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError& e) {
        return app.exit(e);
    }

    if (opts.verbose) {
        spdlog::set_level(spdlog::level::debug);
    } else {
        spdlog::set_level(spdlog::level::info);
    }

    if (opts.force_small && opts.force_large) {
        spdlog::error("Cannot use both --force-small and --force-large");
        return 1;
    }

    try {
        WatermarkEngine engine;

        spdlog::info("Loading: {}", opts.input_path);
        cv::Mat image = cv::imread(opts.input_path, cv::IMREAD_COLOR);
        if (image.empty()) {
            spdlog::error("Failed to load image: {}", opts.input_path);
            return 1;
        }

        spdlog::info("Image: {}x{}", image.cols, image.rows);

        std::optional<WatermarkSize> force_size;
        if (opts.force_small) {
            force_size = WatermarkSize::Small;
        } else if (opts.force_large) {
            force_size = WatermarkSize::Large;
        }

        engine.remove_watermark(image, force_size);

        std::string output = opts.output_path.empty() ? opts.input_path : opts.output_path;
        std::filesystem::path out_path(output);
        if (!out_path.parent_path().empty() && !std::filesystem::exists(out_path.parent_path())) {
            std::filesystem::create_directories(out_path.parent_path());
        }

        std::vector<int> params;
        std::string ext = out_path.extension().string();
        if (ext == ".jpg" || ext == ".jpeg") {
            params = {cv::IMWRITE_JPEG_QUALITY, 100};
        } else if (ext == ".png") {
            params = {cv::IMWRITE_PNG_COMPRESSION, 6};
        } else if (ext == ".webp") {
            params = {cv::IMWRITE_WEBP_QUALITY, 101};
        }

        if (!cv::imwrite(output, image, params)) {
            spdlog::error("Failed to save: {}", output);
            return 1;
        }

        spdlog::info("Saved: {}", output);
        return 0;

    } catch (const std::exception& e) {
        spdlog::error("Error: {}", e.what());
        return 1;
    }
}

} // namespace wmr
