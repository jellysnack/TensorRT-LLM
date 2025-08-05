#include "tensorrt_llm/batch_manager/llguidanceUtil.h"

#include <nlohmann/json.hpp>

namespace tle = tensorrt_llm::executor;

LlgTokenizerPtr llgCreateTokenizer(tle::GuidedDecodingConfig const& guidedDecodingConfig,
                                   tensorrt_llm::runtime::SizeType32 vocabSize)
{
    auto const& tokenizerStr = guidedDecodingConfig.getTokenizerStr();
    TLLM_CHECK_WITH_INFO(tokenizerStr, "missing tokenizerStr");

    TLLM_CHECK_WITH_INFO(guidedDecodingConfig.getStopTokenIds().value().size() == 1, "expected stopTokenIds size to be 1");
    int32_t eosId = guidedDecodingConfig.getStopTokenIds().value().back();

    LlgTokenizerInit tok_init{.vocab_size = static_cast<uint32_t>(vocabSize),
                                .tok_eos = static_cast<uint32_t>(eosId),
                                .tokenizer_json = tokenizerStr.value().c_str(),
                                .use_approximate_greedy_tokenize_fn = true};

    char error_buf[1024];
    auto tokenizer = std::shared_ptr<LlgTokenizer>(
        llg_new_tokenizer(&tok_init, error_buf, sizeof(error_buf)),
        llg_free_tokenizer);
    TLLM_CHECK_WITH_INFO(tokenizer, "error creating tokenizer: %s, error_buf");

    return tokenizer;
}

bool llgValidateGrammar(const tle::GuidedDecodingParams& guidedDecodingParams,
                        LlgTokenizerPtr tokenizer,
                        std::string* errorMessage)
{
    LlgConstraintInit init;
    llg_constraint_init_set_defaults(&init, tokenizer.get());

    int32_t result = 0;
    char error_buf[1024];

    auto guide = guidedDecodingParams.getGuide();
    switch (guidedDecodingParams.getGuideType()) {
        case tle::GuidedDecodingParams::GuideType::kJSON_SCHEMA: {
            auto schema = llgAddSchemaGuidance(guide.value());
            result = llg_validate_grammar(&init, "json_schema", schema.c_str(), error_buf, sizeof(error_buf));
            break;
        }
        case tle::GuidedDecodingParams::GuideType::kREGEX: {
            result = llg_validate_grammar(&init, "regex", guide.value().c_str(), error_buf, sizeof(error_buf));
            break;
        }
        case tle::GuidedDecodingParams::GuideType::kLARK_GRAMMAR: {
            result = llg_validate_grammar(&init, "lark", guide.value().c_str(), error_buf, sizeof(error_buf));
            break;
        }
        case tle::GuidedDecodingParams::GuideType::kJSON: {
            break;
        }
        case tle::GuidedDecodingParams::GuideType::kEBNF_GRAMMAR: {
            TLLM_CHECK_WITH_INFO(false, "kEBNF_GRAMMAR is not supported by the llguidance backend");
        }
    }

    if (result != 0) {
        *errorMessage = std::string(error_buf);
    }

    return result == 0;
}

std::string llgAddSchemaGuidance(const std::string& jsonStr)
{
    TLLM_CHECK_WITH_INFO(!jsonStr.empty(), "empty jsonStr");
    nlohmann::json jsonObj;
    jsonObj = nlohmann::json::parse(jsonStr);

    nlohmann::json guidance;
    guidance["item_separator"] = ", ";
    guidance["key_separator"] = ": ";
    guidance["whitespace_flexible"] = false;
    guidance["lenient"] = true;
    jsonObj["x-guidance"] = guidance;

    return jsonObj.dump();
}
