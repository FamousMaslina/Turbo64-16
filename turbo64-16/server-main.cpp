#include "arg.h"
#include "log.h"
#include "private-logs.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <psapi.h>
#include <shellapi.h>
#endif

int llama_server(int argc, char ** argv);

static void set_default(const char * name, const char * value) {
    if (!std::getenv(name)) {
#ifdef _WIN32
        _putenv_s(name, value);
#else
        setenv(name, value, 0);
#endif
    }
}

static void print_banner() {
    std::fputs(R"(
  _______  _   _  ____   ____    ___
 |__   __|| | | ||  _ \ |  _ \  / _ \
    | |   | | | || |_) || |_) || | | |
    | |   | |_| ||  _ < |  _ < | |_| |
    |_|    \___/ |_| \_\|____/  \___/

 +------------------------------------------------------+
 | TURBO64-16 // PRIVATE LAN INFERENCE SERVER            |
 | Ryzen 7950X | 64 GB DDR5 | NVMe | RTX 4080             |
 | Mapped experts -> bounded read-ahead -> pinned upload |
 +------------------------------------------------------+
)", stderr);
    std::fflush(stderr);
}

struct turbo_profile {
    const char * flag;
    const char * name;
    const char * batch;
    const char * microbatch;
    const char * kv_offload;
    const char * kv_type;
};

static const turbo_profile profiles[] = {
    {"--turbo-prefill",  "prefill",  "16384", "9216", "--no-kv-offload", "f16"},
    {"--turbo-decode",   "decode",   "4096",  "1024", nullptr, nullptr},
    {"--turbo-balanced", "balanced", "4096",  "2048", nullptr, nullptr},
    {"--turbo-prefill2", "prefill2", "4096",  "4096", nullptr, nullptr},
    {"--prefill-kv",     "prefill-kv", "16384", "9216", "--kv-offload", "q8_0"},
};

static void print_profiles() {
    std::fputs("\nTurbo64-16 profiles (select one; native arguments override profile defaults):\n"
               "  --turbo-prefill   Prioritize prompt processing; batch 16384, microbatch 9216, CPU F16 KV.\n"
               "  --turbo-prefill2  Restore the previous prefill settings; batch/microbatch 4096, GPU F16 KV.\n"
               "  --prefill-kv      Use 64K GPU Q8 K/V; batch 16384, microbatch 9216.\n"
               "  --turbo-decode    Prioritize generation; microbatch 1024.\n"
               "  --turbo-balanced Use an intermediate microbatch of 2048.\n"
               "All profiles use the local MiMo model, 64K context, CPU MoE 47, 16/24 threads,\n"
               "Flash Attention, mmap, one slot, Jinja and reasoning. Decode/balanced/prefill2 use batch 4096.\n"
               "No arguments selects --turbo-decode. Custom arguments without a profile retain native defaults.\n\n", stderr);
}

struct hardware_monitor {
    std::mutex mutex;
    std::condition_variable wake;
    bool stopped = false;
    std::thread worker;

    explicit hardware_monitor(bool enabled) {
#ifdef _WIN32
        if (!enabled) {
            return;
        }
        worker = std::thread([this] {
            uint64_t cpu_previous = 0;
            uint64_t read_previous = 0;
            auto time_previous = std::chrono::steady_clock::now();
            const double processors = std::max(1u, std::thread::hardware_concurrency());
            std::unique_lock<std::mutex> lock(mutex);
            while (!stopped) {
                PROCESS_MEMORY_COUNTERS_EX process = {};
                MEMORYSTATUSEX system = {};
                system.dwLength = sizeof(system);
                FILETIME creation, exit, kernel, user;
                IO_COUNTERS io = {};
                if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS *) &process, sizeof(process)) &&
                        GlobalMemoryStatusEx(&system) && GetProcessTimes(GetCurrentProcess(), &creation, &exit, &kernel, &user) &&
                        GetProcessIoCounters(GetCurrentProcess(), &io)) {
                    const uint64_t cpu = ((uint64_t) kernel.dwHighDateTime << 32) + kernel.dwLowDateTime +
                                         ((uint64_t) user.dwHighDateTime << 32) + user.dwLowDateTime;
                    const auto now = std::chrono::steady_clock::now();
                    const double seconds = std::chrono::duration<double>(now - time_previous).count();
                    const double usage = cpu_previous && seconds > 0 ? (cpu - cpu_previous) / 1e7 / seconds / processors * 100 : 0;
                    LOG_INF("Turbo64-16 hardware: cpu=%.1f%% working_set=%.2f GiB private_commit=%.2f GiB peak_working_set=%.2f GiB system_ram_free=%.2f/%.2f GiB ram_load=%lu%% page_faults=%lu logical_read_delta=%.1f MiB\n",
                            usage, process.WorkingSetSize / 1073741824.0, process.PrivateUsage / 1073741824.0,
                            process.PeakWorkingSetSize / 1073741824.0, system.ullAvailPhys / 1073741824.0,
                            system.ullTotalPhys / 1073741824.0, system.dwMemoryLoad, process.PageFaultCount,
                            read_previous ? (io.ReadTransferCount - read_previous) / 1048576.0 : 0);
                    cpu_previous = cpu;
                    read_previous = io.ReadTransferCount;
                    time_previous = now;
                }
                wake.wait_for(lock, std::chrono::seconds(5), [this] { return stopped; });
            }
        });
#else
        (void) enabled;
#endif
    }

    ~hardware_monitor() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stopped = true;
        }
        wake.notify_all();
        if (worker.joinable()) {
            worker.join();
        }
        common_log_flush(common_log_main());
    }
};

int main(int argc, char ** argv) {
    common_log_set_metadata_only(turbo64_16_metadata_log);
#ifdef _WIN32
    // Profile expansion changes argc, so the native parser cannot repair these UTF-8 arguments.
    int count = 0;
    wchar_t ** wide = CommandLineToArgvW(GetCommandLineW(), &count);
    if (!wide) {
        std::fputs("Turbo64-16: cannot read the command line.\n", stderr);
        return 2;
    }
    std::vector<std::string> utf8_arguments;
    for (int i = 0; i < count; ++i) {
        const int size = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, wide[i], -1, nullptr, 0, nullptr, nullptr);
        if (!size) {
            LocalFree(wide);
            std::fputs("Turbo64-16: invalid command-line encoding.\n", stderr);
            return 2;
        }
        std::string value(size, '\0');
        WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, wide[i], -1, value.data(), size, nullptr, nullptr);
        value.pop_back();
        utf8_arguments.push_back(std::move(value));
    }
    LocalFree(wide);
    std::vector<char *> utf8_pointers;
    for (auto & value : utf8_arguments) {
        utf8_pointers.push_back(value.data());
    }
    utf8_pointers.push_back(nullptr);
    argc = count;
    argv = utf8_pointers.data();
#endif
    common_params params;
    auto parser = common_params_parser_init(params, LLAMA_EXAMPLE_SERVER);
    const turbo_profile * profile = argc == 1 ? &profiles[1] : nullptr;
    std::vector<std::string> supplied;
    std::set<std::string> specified;
    bool information = false;
    bool help = false;
    for (int i = 1; i < argc; ++i) {
        std::string option = argv[i];
        std::replace(option.begin(), option.end(), '_', '-');
        if (option == "--log-prompts-dir" || option == "--verbose-prompt") {
            std::fputs("Turbo64-16: conversation-content capture options are disabled.\n", stderr);
            return 2;
        }
        const turbo_profile * selected = nullptr;
        for (const auto & candidate : profiles) {
            if (option == candidate.flag) {
                selected = &candidate;
                break;
            }
        }
        if (selected) {
            if (profile) {
                std::fputs("Turbo64-16: select only one Turbo profile.\n", stderr);
                return 2;
            }
            profile = selected;
            continue;
        }
        help |= option == "--help" || option == "-h";
        information |= help || option == "--version" || option == "--list-devices";
        supplied.emplace_back(argv[i]);
        // Use the native option metadata so argument values are never read as profile flags.
        for (const auto & opt : parser.options) {
            if (std::find(opt.args.begin(), opt.args.end(), option) == opt.args.end() &&
                    std::find(opt.args_neg.begin(), opt.args_neg.end(), option) == opt.args_neg.end()) {
                continue;
            }
            specified.insert(opt.args.begin(), opt.args.end());
            specified.insert(opt.args_neg.begin(), opt.args_neg.end());
            const int values = opt.handler_str_str ? 2 : (opt.handler_int || opt.handler_string ? 1 : 0);
            for (int value = 0; value < values && i + 1 < argc; ++value) {
                supplied.emplace_back(argv[++i]);
            }
            break;
        }
    }
    print_banner();
    if (help) {
        print_profiles();
    }
    set_default("TURBO64_16", "2");
    set_default("TURBO64_16_DIAGNOSTICS", "1");
    const bool diagnostics = std::strcmp(std::getenv("TURBO64_16_DIAGNOSTICS"), "1") == 0;
    set_default("LLAMA_ARG_HOST", "0.0.0.0");
    set_default("LLAMA_ARG_PORT", "5559");
    set_default("LLAMA_ARG_LOG_VERBOSITY", diagnostics ? "5" : "3");
    set_default("LLAMA_ARG_LOG_PREFIX", "1");
    set_default("LLAMA_ARG_LOG_TIMESTAMPS", "1");
    set_default("LLAMA_ARG_ENDPOINT_METRICS", "1");
    set_default("LLAMA_ARG_ENDPOINT_SLOTS", "1");
    set_default("LLAMA_ARG_PERF", "1");

    if (!information && diagnostics && !std::getenv("LLAMA_ARG_LOG_FILE")) {
        std::error_code error;
        const auto directory = std::filesystem::absolute(argv[0]).parent_path() / "logs";
        std::filesystem::create_directories(directory, error);
        if (!error) {
            const auto stamp = std::chrono::system_clock::now().time_since_epoch().count();
            const auto file = (directory / ("turbo-" + std::to_string(stamp) + ".log")).string();
            set_default("LLAMA_ARG_LOG_FILE", file.c_str());
            std::fprintf(stderr, "Turbo64-16 diagnostic log: %s\n", file.c_str());
        }
    }

    std::vector<std::string> arguments = {argv[0]};
    if (profile && !information) {
        const char * defaults[][2] = {
            {"-m", "C:/Users/Tudi/Documents/Tudi/AI/MiMo-V2.6-Flash-MOPD-Q2_K-00001-of-00002.gguf"},
            {"-c", "65536"}, {"-ngl", "all"}, {"--n-cpu-moe", "47"}, {"-t", "16"}, {"-tb", "24"},
            {"-b", profile->batch}, {"-ub", profile->microbatch}, {"-fa", "on"}, {"-lm", "mmap"},
            {"-np", "1"}, {"--reasoning", "on"},
        };
        for (const auto & setting : defaults) {
            if (specified.count(setting[0])) {
                continue;
            }
            arguments.emplace_back(setting[0]);
            arguments.emplace_back(setting[1]);
        }
        if (profile->kv_offload && !specified.count(profile->kv_offload)) {
            arguments.emplace_back(profile->kv_offload);
        }
        if (profile->kv_type) {
            for (const char * option : {"-ctk", "-ctv"}) {
                if (!specified.count(option)) {
                    arguments.emplace_back(option);
                    arguments.emplace_back(profile->kv_type);
                }
            }
        }
        set_default("TURBO64_16_READAHEAD_ASYNC", "1");
        set_default("TURBO64_16_READAHEAD_MIB", "128");
        set_default("LLAMA_ARG_JINJA", "1");
        std::fprintf(stderr, "Turbo profile=%s | native CLI overrides are applied after profile defaults.\n", profile->name);
    }
    arguments.insert(arguments.end(), supplied.begin(), supplied.end());
    std::vector<char *> pointers;
    for (auto & argument : arguments) {
        pointers.push_back(argument.data());
    }
    pointers.push_back(nullptr);
    if (!information) {
        std::fprintf(stderr, "Turbo mode=%s diagnostics=%s | OpenAI API: http://<PC-LAN-IP>:5559/v1 (default bind)\n",
                     std::getenv("TURBO64_16"), diagnostics ? "on" : "off");
        std::fputs("Monitoring: GET /metrics, GET /slots, GET /props | CLI arguments override server defaults.\n", stderr);
    }
    hardware_monitor monitor(diagnostics && !information);
    return llama_server((int) arguments.size(), pointers.data());
}
