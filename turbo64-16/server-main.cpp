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
#include <string>
#include <thread>
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
    for (int i = 1; i < argc; ++i) {
        std::string option = argv[i];
        std::replace(option.begin(), option.end(), '_', '-');
        if (option == "--log-prompts-dir" || option == "--verbose-prompt") {
            std::fputs("Turbo64-16: conversation-content capture options are disabled.\n", stderr);
            return 2;
        }
    }
    print_banner();
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

    bool information = false;
    for (int i = 1; i < argc; ++i) {
        information |= std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0 ||
                       std::strcmp(argv[i], "--version") == 0 || std::strcmp(argv[i], "--list-devices") == 0;
    }
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

    std::vector<std::string> arguments;
    if (argc == 1) {
        arguments = {argv[0], "-m", "C:/Users/Tudi/Documents/Tudi/AI/MiMo-V2.6-Flash-MOPD-Q2_K-00001-of-00002.gguf",
                     "-c", "65536", "-ngl", "all", "--n-cpu-moe", "47", "-t", "16", "-tb", "24",
                     "-b", "4096", "-ub", "1024", "-fa", "on", "-lm", "mmap", "-np", "1", "--jinja", "--reasoning", "on"};
    } else {
        for (int i = 0; i < argc; ++i) {
            arguments.emplace_back(argv[i]);
        }
    }
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
