#include "private-logs.h"

#include <cstring>

static bool starts_with(const char * text, const char * prefix) {
    return std::strncmp(text, prefix, std::strlen(prefix)) == 0;
}

bool turbo64_16_metadata_log(const char * message, bool formatted) {
    const char * turbo_records[] = {
        "Turbo64-16 hardware:", "Turbo64-16 expert:", "Turbo64-16 graph:",
        "Turbo64-16: whole-model mmap prefetch disabled", "Turbo64-16: staging allocation failed",
        "Turbo64-16: %s expert uploads", "Turbo64-16: CUDA",
    };
    for (const char * prefix : turbo_records) {
        if (starts_with(message, prefix)) {
            return true;
        }
    }

    if (formatted) {
        const char * backend_records[] = {
            "ggml_cuda_init:", "  Device ", "load_tensors:", "sched_reserve:",
            "llama_context:", "llama_kv_cache:", "llama_kv_cache_iswa:",
            "llama_perf_context_print:", "llama_perf_sampler_print:",
        };
        for (const char * prefix : backend_records) {
            if (starts_with(message, prefix)) {
                return true;
            }
        }
        return false;
    }

    const char * function_prefixes[] = {
        "srv  %12.*s: ", "slot %12.*s: id %2d | task %d | ", "%s: ",
    };
    for (const char * prefix : function_prefixes) {
        if (starts_with(message, prefix)) {
            message += std::strlen(prefix);
            break;
        }
    }
    while (*message == ' ') {
        ++message;
    }
    const char * counters[] = {
        "HTTP metadata:", "prompt processing, n_tokens =", "prompt eval time =",
        "eval time =", "total time =", "load time =", "sampling time =", "samplers time =",
        "unaccounted time =", "graphs reused =", "n_gen = %6d, tg =", "n_batch (effective) =",
        "created context checkpoint ",
    };
    bool counter = false;
    for (const char * prefix : counters) {
        counter |= starts_with(message, prefix);
    }
    if (!counter) {
        return false;
    }

    // Counters may interpolate numbers, but never text or individual token characters.
    for (const char * p = message; *p; ++p) {
        if (*p != '%') {
            continue;
        }
        ++p;
        if (*p == '%') {
            continue;
        }
        while (*p && std::strchr("-+ #0.123456789*hljztLI", *p)) {
            ++p;
        }
        if (!*p || !std::strchr("diuoxXaAeEfFgGp", *p)) {
            return false;
        }
    }
    return true;
}
