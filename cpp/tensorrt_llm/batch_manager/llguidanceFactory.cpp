#include "tensorrt_llm/batch_manager/llguidanceFactory.h"

namespace tle = tensorrt_llm::executor;

namespace
{

void checkMatcher(LlgMatcher* matcher)
{
    char const* matcher_error = llg_matcher_get_error(matcher);
    TLLM_CHECK_WITH_INFO(matcher_error == nullptr, "matcher error: %s", matcher_error);
}

} // namespace

namespace tensorrt_llm::batch_manager
{
LLGuidanceMatcher::LLGuidanceMatcher(LlgMatcherPtr matcher)
    : mMatcher(matcher)
{
}

bool LLGuidanceMatcher::AcceptToken(int32_t tokenId)
{
    if (llg_matcher_consume_token(mMatcher.get(), tokenId) != 0)
    {
        checkMatcher(mMatcher.get());
        return false;
    }
    return true;
}

void LLGuidanceMatcher::FillNextTokenBitmask(DLTensor* nextTokenBitmask)
{
    uint32_t* mask = static_cast<uint32_t*>(nextTokenBitmask->data);
    TLLM_CHECK_WITH_INFO(nextTokenBitmask->ndim == 1, "expected nextTokenBitmask ndim to be 1");
    size_t mask_byte_len = nextTokenBitmask->shape[0] * sizeof(int32_t);

    if (llg_matcher_compute_mask_into(mMatcher.get(), mask, mask_byte_len) != 0)
    {
        checkMatcher(mMatcher.get());
    }
}

LLGuidanceMatcherFactory::LLGuidanceMatcherFactory(
    tle::GuidedDecodingConfig const& guidedDecodingConfig, const SizeType32 vocabSize)
{
    TLLM_LOG_INFO("Using LLGuidance for Structured Output");
    mTokenizer = llgCreateTokenizer(guidedDecodingConfig, vocabSize);
}

std::shared_ptr<IGrammarMatcher> LLGuidanceMatcherFactory::Create(
    tle::GuidedDecodingParams::GuideType guideType, std::optional<std::string> guide)
{
    LlgMatcherPtr matcher;
    LlgConstraintInit init;
    llg_constraint_init_set_defaults(&init, mTokenizer.get());
    // have to set a larger fuel value to process many parallel requests, not sure how it works
    init.limits.initial_lexer_fuel = 100000000;
    init.limits.step_lexer_fuel = 100000000;

    switch (guideType)
    {
    case tle::GuidedDecodingParams::GuideType::kJSON:
    {
        auto schema = llgAddSchemaGuidance("{\"type\": \"object\"}");
        matcher = std::shared_ptr<LlgMatcher>(llg_new_matcher(&init, "json_schema", schema.c_str()), llg_free_matcher);
        break;
    }
    case tle::GuidedDecodingParams::GuideType::kJSON_SCHEMA:
    {
        auto schema = llgAddSchemaGuidance(guide.value());
        matcher = std::shared_ptr<LlgMatcher>(llg_new_matcher(&init, "json_schema", schema.c_str()), llg_free_matcher);
        break;
    }
    case tle::GuidedDecodingParams::GuideType::kREGEX:
    {
        matcher = std::shared_ptr<LlgMatcher>(llg_new_matcher(&init, "regex", guide.value().c_str()), llg_free_matcher);
        break;
    }
    case tle::GuidedDecodingParams::GuideType::kLARK_GRAMMAR:
    {
        matcher = std::shared_ptr<LlgMatcher>(llg_new_matcher(&init, "lark", guide.value().c_str()), llg_free_matcher);
        break;
    }
    case tle::GuidedDecodingParams::GuideType::kEBNF_GRAMMAR:
    {
        TLLM_CHECK_WITH_INFO(false, "kEBNF_GRAMMAR is not supported by the llguidance backend");
    }
    case tle::GuidedDecodingParams::GuideType::kSTRUCTURAL_TAG:
    {
        TLLM_CHECK_WITH_INFO(false, "kSTRUCTURAL_TAG is not supported by the llguidance backend");
    }
    }

    checkMatcher(matcher.get());

    return std::make_shared<LLGuidanceMatcher>(matcher);
}

} // namespace tensorrt_llm::batch_manager
