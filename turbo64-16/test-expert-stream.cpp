#include "expert-stream.h"

#include "ggml-backend.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>

int main() {
    if (!turbo64_16_mode()) {
        std::fprintf(stderr, "Set TURBO64_16=1 or 2 before running this test.\n");
        return 1;
    }
    ggml_backend_load_all();
    auto * dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!dev) {
        std::fprintf(stderr, "A GPU backend is required.\n");
        return 1;
    }
    ggml_backend_ptr backend(ggml_backend_dev_init(dev, nullptr));
    if (!backend) {
        return 1;
    }
    ggml_init_params params = { 2 * ggml_tensor_overhead(), nullptr, true };
    for (ggml_type type : { GGML_TYPE_Q2_K, GGML_TYPE_MXFP4 }) {
        ggml_context_ptr ctx(ggml_init(params));
        auto * src = ggml_new_tensor_3d(ctx.get(), type, 4096, 2048, 64);
        auto * dst = ggml_dup_tensor(ctx.get(), src);
        const size_t size = ggml_nbytes(src);
        std::vector<uint8_t> source(size);
        std::vector<uint8_t> expected(size);
        std::vector<uint8_t> actual(size);
        for (size_t i = 0; i < size; ++i) {
            source[i] = (uint8_t) (i * 37 + i / 4096);
        }
        src->data = source.data();
        ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
        if (!buffer) {
            return 1;
        }

        for (int pattern = 0; pattern < 5; ++pattern) {
            std::vector<bool> used(64);
            for (size_t i = 0; i < used.size(); ++i) {
                used[i] = pattern == 0 || (pattern == 1 && i % 2 == 0) ||
                    (pattern == 2 && i >= 11 && i < 54) || (pattern == 3 && i == 63);
            }
            std::fill(expected.begin(), expected.end(), 0xcc);
            ggml_backend_tensor_set(dst, expected.data(), 0, size);
            for (size_t i = 0; i < used.size(); ++i) {
                if (used[i]) {
                    const size_t offset = i * src->nb[2];
                    const size_t length = src->nb[2] + (i + 1 < used.size() ? 512 : 0);
                    std::memcpy(expected.data() + offset, source.data() + offset, length);
                }
            }
            {
                turbo64_16_stream stream;
                if (stream.copy(backend.get(), src, dst, used, 1)) {
                    std::fprintf(stderr, "Decode must use the original copy path.\n");
                    return 1;
                }
                // Repeat without synchronizing to exercise event-protected buffer reuse.
                if (!stream.copy(backend.get(), src, dst, used, 1024) ||
                        !stream.copy(backend.get(), src, dst, used, 1024)) {
                    return 1;
                }
            }
            ggml_backend_tensor_get(dst, actual.data(), 0, size);
            if (actual != expected) {
                std::fprintf(stderr, "%s pattern %d: byte mismatch\n", ggml_type_name(type), pattern);
                return 1;
            }
            std::printf("%s pattern %d: exact bytes, padding, reuse and destruction passed\n", ggml_type_name(type), pattern);
        }
    }
    return 0;
}
