#include "tensorrt_llm/batch_manager/xgrammarFactory.h"

#include <xgrammar/xgrammar.h>
#include <picojson.h>

namespace tle = tensorrt_llm::executor;

namespace {

template<typename T>
T parseEnvVar(const char* envVal, T defaultValue) {
    if (!envVal) {
        return defaultValue;
    }

    std::istringstream stream(envVal);
    T parsedValue;
    stream >> parsedValue;

    TLLM_CHECK_WITH_INFO(!stream.fail() && stream.eof(), "Failed to parse environment variable value '%s'", envVal);

    return parsedValue;
}

} // namespace

namespace tensorrt_llm::batch_manager
{
    XGrammarMatcher::XGrammarMatcher(std::shared_ptr<xgrammar::GrammarMatcher> grammarMatcher)
        : mGrammarMatcher(grammarMatcher)
    {
    }

    bool XGrammarMatcher::AcceptToken(int32_t tokenId)
    {
        return mGrammarMatcher->AcceptToken(tokenId);
    }

    void XGrammarMatcher::FillNextTokenBitmask(DLTensor* nextTokenBitmask)
    {
        mGrammarMatcher->FillNextTokenBitmask(nextTokenBitmask);
    }

    XGrammarMatcherFactory::XGrammarMatcherFactory(tle::GuidedDecodingConfig const& guidedDecodingConfig,
                                                   const SizeType32 vocabSizePadded)
    {
        TLLM_LOG_INFO("Using XGrammar for Structured Output");

        const int max_threads = parseEnvVar(std::getenv("XGR_MAX_THREADS"), 8);
        const bool cache_enabled = parseEnvVar(std::getenv("XGR_CACHE_ENABLED"), false);
        const long long max_mmemory_bytes = parseEnvVar(std::getenv("XGR_MAX_MEMORY_BYTES"), -1);

        TLLM_LOG_INFO("[XGrammarCompiler] max_threads: %d", max_threads);
        TLLM_LOG_INFO("[XGrammarCompiler] cache_enabled: %s", cache_enabled ? "true" : "false");
        TLLM_LOG_INFO("[XGrammarCompiler] max_mmemory_bytes: %lld", max_mmemory_bytes);

        auto const& tokenizerStr = guidedDecodingConfig.getTokenizerStr();
        if (tokenizerStr)
        {
            auto const& metadata = xgrammar::TokenizerInfo::DetectMetadataFromHF(tokenizerStr.value());
            picojson::value v;
            std::string err = picojson::parse(v, metadata);
            TLLM_CHECK_WITH_INFO(err.empty(), "Failed to parse metadata: %s", err.c_str());

            const picojson::object& obj = v.get<picojson::object>();
            TLLM_CHECK_WITH_INFO(obj.count("vocab_type") && obj["vocab_type"].is<std::int64_t>(),
                "Missing or invalid 'vocab_type' in metadata");
            int vocab_type_int = static_cast<int>(obj["vocab_type"].get<int64_t>());
            TLLM_CHECK_WITH_INFO(vocab_type_int == 0 || vocab_type_int == 1 || vocab_type_int == 2,
                "Invalid vocab_type in metadata: %d", vocab_type_int);
            xgrammar::VocabType vocab_type = static_cast<xgrammar::VocabType>(vocab_type_int);

            TLLM_CHECK_WITH_INFO(obj.count("add_prefix_space") && obj["add_prefix_space"].is<bool>(),
                "Missing or invalid 'add_prefix_space' in metadata");
            bool add_prefix_space = obj["add_prefix_space"].get<bool>();

            auto const& tokenizerInfo = xgrammar::TokenizerInfo(guidedDecodingConfig.getEncodedVocab().value(),
                vocab_type, vocabSizePadded, guidedDecodingConfig.getStopTokenIds(), add_prefix_space);
            mXGrammarCompiler = std::make_shared<xgrammar::GrammarCompiler>(
                tokenizerInfo, max_threads, cache_enabled, max_mmemory_bytes);
        }
        else
        {
            auto const& tokenizerInfo = xgrammar::TokenizerInfo(guidedDecodingConfig.getEncodedVocab().value(),
                xgrammar::VocabType::RAW, vocabSizePadded, guidedDecodingConfig.getStopTokenIds());
            mXGrammarCompiler = std::make_shared<xgrammar::GrammarCompiler>(
                tokenizerInfo, max_threads, cache_enabled, max_mmemory_bytes);
        }
    }

    std::shared_ptr<IGrammarMatcher> XGrammarMatcherFactory::Create(tle::GuidedDecodingParams::GuideType guideType,
                                                                    std::optional<std::string> guide)
    {
        std::shared_ptr<xgrammar::GrammarMatcher> grammarMatcher;

        switch (guideType)
        {
            case tle::GuidedDecodingParams::GuideType::kJSON:
            {
                grammarMatcher = std::make_shared<xgrammar::GrammarMatcher>(
                    mXGrammarCompiler->CompileBuiltinJSONGrammar());
                break;
            }
            case tle::GuidedDecodingParams::GuideType::kJSON_SCHEMA:
            {
                grammarMatcher = std::make_shared<xgrammar::GrammarMatcher>(
                    mXGrammarCompiler->CompileJSONSchema(guide.value(), /*any_whitespace*/ false));
                break;
            }
            case tle::GuidedDecodingParams::GuideType::kREGEX:
            {
                auto const& grammar = xgrammar::Grammar::FromRegex(guide.value());
                grammarMatcher = std::make_shared<xgrammar::GrammarMatcher>(
                    mXGrammarCompiler->CompileGrammar(grammar));
                break;
            }
            case tle::GuidedDecodingParams::GuideType::kEBNF_GRAMMAR:
            {
                auto const& grammar = xgrammar::Grammar::FromEBNF(guide.value());
                grammarMatcher = std::make_shared<xgrammar::GrammarMatcher>(mXGrammarCompiler->CompileGrammar(grammar));
                break;
            }
        }

        return std::make_shared<XGrammarMatcher>(grammarMatcher);
    }
} // namespace tensorrt_llm::batch_manager
