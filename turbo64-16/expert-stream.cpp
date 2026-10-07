#include "expert-stream.h"

#include "llama-impl.h"
#include "llama-mmap.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <future>

static constexpr size_t stage_size = 32 * 1024 * 1024;
static constexpr size_t prefetch_size = 128 * 1024 * 1024;

static size_t read_ahead_size() {
    static const size_t size = [] {
        const char * value = std::getenv("TURBO64_16_READAHEAD_MIB");
        if (value) {
            char * end = nullptr;
            const unsigned long mib = std::strtoul(value, &end, 10);
            if (*value && !*end && mib >= 32 && mib <= 256) {
                return (size_t) mib * 1024 * 1024;
            }
        }
        return prefetch_size;
    }();
    return size;
}

static bool read_ahead_async() {
    static const bool enabled = [] {
        const char * value = std::getenv("TURBO64_16_READAHEAD_ASYNC");
        return !value || std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

static bool diagnostics_enabled() {
    static const bool enabled = [] {
        const char * value = std::getenv("TURBO64_16_DIAGNOSTICS");
        return value && std::strcmp(value, "1") == 0;
    }();
    return enabled;
}

void turbo64_16_stream::begin_graph() {
    if (diagnostics_enabled()) {
        stats = {};
        graph_start_us = ggml_time_us();
    }
}

void turbo64_16_stream::end_graph(ggml_status status) {
    if (diagnostics_enabled()) {
        LLAMA_LOG_INFO("Turbo64-16 graph: status=%d submit_ms=%.2f expert_tensors=%zu selected_matrices=%zu upload_requested=%.2f MiB chunks=%zu prefetch_ms=%.2f prefetch_wait_ms=%.2f host_stage_ms=%.2f buffer_wait_ms=%.2f enqueue_ms=%.2f\n",
                (int) status, (ggml_time_us() - graph_start_us) / 1000.0, stats.tensors, stats.experts,
                stats.bytes / 1048576.0, stats.chunks, stats.prefetch_us / 1000.0, stats.prefetch_wait_us / 1000.0,
                stats.stage_us / 1000.0, stats.wait_us / 1000.0, stats.enqueue_us / 1000.0);
    }
}

int turbo64_16_mode() {
    static const int mode = [] {
#ifdef _WIN32
        const char * value = std::getenv("TURBO64_16");
        if (value && std::strcmp(value, "1") == 0) {
            return 1;
        }
        if (value && std::strcmp(value, "2") == 0) {
            return 2;
        }
#endif
        return 0;
    }();
    return mode;
}

turbo64_16_stream::~turbo64_16_stream() {
    for (auto & s : slots) {
        if (s.pending) {
            ggml_backend_event_synchronize(s.event.get());
        }
    }
}

bool turbo64_16_stream::init(ggml_backend_t backend) {
    auto * dev = ggml_backend_get_device(backend);
    if (attempted) {
        return device == dev;
    }
    attempted = true;

    ggml_backend_dev_props props;
    ggml_backend_dev_get_props(dev, &props);
    if (!props.caps.async || !props.caps.host_buffer || !props.caps.events) {
        return false;
    }
    auto * buft = ggml_backend_dev_host_buffer_type(dev);
    if (!buft) {
        return false;
    }

    for (auto & s : slots) {
        s.buffer.reset(ggml_backend_buft_alloc_buffer(buft, stage_size));
        s.event.reset(ggml_backend_event_new(dev));
        if (!s.buffer || !s.event || ggml_backend_buffer_get_type(s.buffer.get()) != buft) {
            for (auto & allocated : slots) {
                allocated.event.reset();
                allocated.buffer.reset();
            }
            LLAMA_LOG_WARN("Turbo64-16: staging allocation failed, using prefetch only\n");
            return false;
        }
    }
    device = dev;
    LLAMA_LOG_INFO("Turbo64-16: %s expert uploads, 2 x 32 MiB host buffers, %zu MiB read-ahead, async=%d\n", ggml_backend_name(backend), read_ahead_size() / 1048576, (int) read_ahead_async());
    return true;
}

bool turbo64_16_stream::copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, const std::vector<bool> & used, int64_t n_tokens) {
    const int mode = turbo64_16_mode();
    if (!mode || n_tokens < 32 || !src->data || !ggml_is_contiguous(src) || !ggml_are_same_shape(src, dst) ||
            std::memcmp(src->nb, dst->nb, sizeof(src->nb)) != 0 ||
            src->ne[2] <= 0 || src->ne[3] != 1 || used.size() != (size_t) src->ne[2]) {
        return false;
    }

    const bool staging = mode == 2 && init(backend);
    const bool diagnostics = diagnostics_enabled();
    const int64_t started = diagnostics ? ggml_time_us() : 0;
    transfer_stats current;
    const size_t expert_size = src->nb[2];
    const size_t tensor_size = ggml_nbytes(src);
    if (!expert_size || used.size() > tensor_size / expert_size) {
        return false;
    }

    struct transfer_range {
        size_t offset;
        size_t length;
    };
    std::vector<std::vector<transfer_range>> windows;
    size_t window_bytes = 0;
    const size_t window_limit = read_ahead_size();

    for (size_t first = 0; first < used.size();) {
        if (!used[first]) {
            ++first;
            continue;
        }
        size_t last = first + 1;
        while (last < used.size() && used[last]) {
            ++last;
        }

        // Preserve the scheduler's MMQ padding at the end of each expert run.
        const size_t offset = first * expert_size;
        const size_t padding = last < used.size() ? std::min<size_t>(expert_size, 512) : 0;
        const size_t length = (last - first) * expert_size + padding;
        GGML_ASSERT(offset <= tensor_size && length <= tensor_size - offset);
        if (diagnostics) {
            current.experts += last - first;
            current.bytes += length;
        }

        for (size_t pos = 0; pos < length;) {
            if (windows.empty() || window_bytes == window_limit) {
                windows.emplace_back();
                window_bytes = 0;
            }
            const size_t size = std::min(window_limit - window_bytes, length - pos);
            windows.back().push_back({offset + pos, size});
            window_bytes += size;
            pos += size;
        }
        first = last;
    }

    auto prefetch_window = [&](size_t index) {
        llama_memory_ranges ranges;
        for (const auto & range : windows[index]) {
            ranges.push_back({(const uint8_t *) src->data + range.offset, range.length});
        }
        const int64_t before = diagnostics ? ggml_time_us() : 0;
        llama_prefetch(std::move(ranges));
        return diagnostics ? ggml_time_us() - before : 0;
    };
    const bool async = staging && read_ahead_async();
    std::future<int64_t> pending;
    if (async && !windows.empty()) {
        pending = std::async(std::launch::async | std::launch::deferred, prefetch_window, 0);
    }
    for (size_t index = 0; index < windows.size(); ++index) {
        if (async) {
            const int64_t before = diagnostics ? ggml_time_us() : 0;
            current.prefetch_us += pending.get();
            if (diagnostics) {
                current.prefetch_wait_us += ggml_time_us() - before;
            }
            // Only one future window can be in flight; all tasks finish before src is released.
            if (index + 1 < windows.size()) {
                pending = std::async(std::launch::async | std::launch::deferred, prefetch_window, index + 1);
            }
        } else {
            current.prefetch_us += prefetch_window(index);
        }

        for (const auto & range : windows[index]) {
            for (size_t chunk = 0; chunk < range.length;) {
                const size_t size = std::min(stage_size, range.length - chunk);
                const size_t target = range.offset + chunk;
                const void * data = (const uint8_t *) src->data + target;
                int64_t before;
                if (staging) {
                    auto & s = slots[next_slot];
                    if (s.pending) {
                        before = diagnostics ? ggml_time_us() : 0;
                        ggml_backend_event_synchronize(s.event.get());
                        if (diagnostics) {
                            current.wait_us += ggml_time_us() - before;
                        }
                    }
                    void * host = ggml_backend_buffer_get_base(s.buffer.get());
                    before = diagnostics ? ggml_time_us() : 0;
                    std::memcpy(host, data, size);
                    if (diagnostics) {
                        current.stage_us += ggml_time_us() - before;
                    }
                    before = diagnostics ? ggml_time_us() : 0;
                    ggml_backend_tensor_set_async(backend, dst, host, target, size);
                    ggml_backend_event_record(s.event.get(), backend);
                    s.pending = true;
                    next_slot = (next_slot + 1) % slots.size();
                } else {
                    before = diagnostics ? ggml_time_us() : 0;
                    ggml_backend_tensor_set_async(backend, dst, data, target, size);
                }
                if (diagnostics) {
                    ++current.chunks;
                    current.enqueue_us += ggml_time_us() - before;
                }
                chunk += size;
            }
        }
    }
    if (diagnostics) {
        ++stats.tensors;
        stats.experts += current.experts;
        stats.bytes += current.bytes;
        stats.chunks += current.chunks;
        stats.prefetch_us += current.prefetch_us;
        stats.prefetch_wait_us += current.prefetch_wait_us;
        stats.wait_us += current.wait_us;
        stats.stage_us += current.stage_us;
        stats.enqueue_us += current.enqueue_us;
        LLAMA_LOG_INFO("Turbo64-16 expert: tensor=%s type=%s backend=%s tokens=%lld selected=%zu/%zu requested=%.2f MiB pinned=%s chunks=%zu host_ms=%.2f prefetch_ms=%.2f prefetch_wait_ms=%.2f stage_ms=%.2f wait_ms=%.2f enqueue_ms=%.2f\n",
                src->name, ggml_type_name(src->type), ggml_backend_name(backend), (long long) n_tokens,
                current.experts, used.size(), current.bytes / 1048576.0, staging ? "yes" : "no", current.chunks,
                (ggml_time_us() - started) / 1000.0, current.prefetch_us / 1000.0, current.prefetch_wait_us / 1000.0,
                current.stage_us / 1000.0, current.wait_us / 1000.0, current.enqueue_us / 1000.0);
    }
    return true;
}
