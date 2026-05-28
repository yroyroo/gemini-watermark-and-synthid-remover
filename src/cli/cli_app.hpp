#pragma once

#include <string>

namespace wmr {

struct CliOptions {
    std::string input_path;
    std::string output_path;
    bool force = false;
    bool force_small = false;
    bool force_large = false;
    bool verbose = false;
};

int run_cli(int argc, char* argv[]);

} // namespace wmr
