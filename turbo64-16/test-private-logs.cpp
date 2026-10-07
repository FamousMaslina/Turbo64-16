#include "private-logs.h"

#include "chat.h"
#include "json.h"
#include "log.h"

#include <climits>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>

int main(int argc, char ** argv) {
    if (argc != 2) {
        return 1;
    }
    const std::string marker = "TURBO_PRIVATE_CONTENT_CANARY_73d6";
    if (common_log_is_metadata_only()) {
        return 1;
    }
    common_log_set_metadata_only(turbo64_16_metadata_log);
    common_log_set_verbosity_thold(INT_MAX);
    common_log_set_file(common_log_main(), argv[1]);

    LOG_DBG("Parsed message: %s\n", marker.c_str());
    LOG_DBG("srv  %12.*s: request: %s\n", 12, __func__, marker.c_str());
    LOG_DBG("srv  %12.*s: response: %s\n", 12, __func__, marker.c_str());
    LOG_WRN("%s: unparsed output: %s\n", __func__, marker.c_str());
    LOG_ERR("%s: error: %s\n", __func__, marker.c_str());
    LOG_DBG("%s: prefill token: %d = %s\n", __func__, 42, marker.c_str());
    LOG_DBG("slot %12.*s: id %2d | task %d | slot decode token, id=%d\n", 12, __func__, 0, 1, 42);
    common_log_default_callback(GGML_LOG_LEVEL_ERROR, marker.c_str(), nullptr);
    common_log_set_jsonl(true);
    common_log_add_json(common_log_main(), "request", common_json{{"content", marker}});
    LOG_INF("Turbo64-16 hardware: cpu=%.1f%%\n", 1.0);
    LOG_TRC("srv  %12.*s: HTTP metadata: status=%d request_bytes=%zu response_bytes=%zu\n", 12, __func__, 200, size_t(10), size_t(20));
    LOG_INF("slot %12.*s: id %2d | task %d | prompt eval time = %.2f ms / %d tokens\n", 12, __func__, 0, 1, 10.0, 32);
    common_log_default_callback(GGML_LOG_LEVEL_INFO, "Turbo64-16 graph: status=0 expert_tensors=138\n", nullptr);
    LOG_INF("slot %12.*s: id %2d | task %d | prompt eval time = %s\n", 12, __func__, 0, 1, marker.c_str());

    common_chat_parser_params params;
    params.debug = true;
    const auto msg = common_chat_peg_parse(common_peg_arena{}, marker, false, params);
    if (msg.content != marker) {
        return 1;
    }
    common_log_flush(common_log_main());
    common_log_set_file(common_log_main(), nullptr);
    std::ifstream file(argv[1]);
    const std::string contents((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    if (contents.find(marker) != std::string::npos || contents.find("Parsed message") != std::string::npos ||
            contents.find("slot decode token") != std::string::npos || contents.find("Turbo64-16 hardware") == std::string::npos ||
            contents.find("prompt eval time") == std::string::npos || contents.find("Turbo64-16 graph") == std::string::npos ||
            contents.find("HTTP metadata:") == std::string::npos) {
        std::fputs("FAIL: private logging policy\n", stderr);
        return 1;
    }
    std::puts("PASS: content, token traces, parser AST and structured dumps blocked; technical counters retained.");
    return 0;
}
