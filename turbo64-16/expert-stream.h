#pragma once

#include "ggml-cpp.h"

#include <array>
#include <vector>

int turbo64_16_mode();

struct turbo64_16_stream {
    ~turbo64_16_stream();

    bool copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, const std::vector<bool> & used, int64_t n_tokens);
    void begin_graph();
    void end_graph(ggml_status status);

private:
    struct slot {
        ggml_backend_buffer_ptr buffer;
        ggml_backend_event_ptr event;
        bool pending = false;
    };

    bool init(ggml_backend_t backend);

    std::array<slot, 2> slots;
    ggml_backend_dev_t device = nullptr;
    size_t next_slot = 0;
    bool attempted = false;

    struct transfer_stats {
        size_t tensors = 0;
        size_t experts = 0;
        size_t bytes = 0;
        size_t chunks = 0;
        int64_t prefetch_us = 0;
        int64_t wait_us = 0;
        int64_t stage_us = 0;
        int64_t enqueue_us = 0;
    } stats;
    int64_t graph_start_us = 0;
};
