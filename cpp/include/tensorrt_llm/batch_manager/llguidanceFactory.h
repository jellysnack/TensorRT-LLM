#include "llguidanceUtil.h"
#include "tensorrt_llm/batch_manager/guidedDecoder.h"

namespace tensorrt_llm::batch_manager
{
class LLGuidanceMatcher final : public IGrammarMatcher
{
public:
    explicit LLGuidanceMatcher(LlgMatcherPtr matcher);

    bool AcceptToken(int32_t tokenId) override;
    void FillNextTokenBitmask(DLTensor* nextTokenBitmask) override;

private:
    LlgMatcherPtr mMatcher;
};

class LLGuidanceMatcherFactory final : public IGrammarMatcherFactory
{
public:
    using SizeType32 = tensorrt_llm::runtime::SizeType32;

    LLGuidanceMatcherFactory(
        tensorrt_llm::executor::GuidedDecodingConfig const& guidedDecodingConfig, SizeType32 vocabSize);

    std::shared_ptr<IGrammarMatcher> Create(tensorrt_llm::executor::GuidedDecodingParams::GuideType guideType,
        std::optional<std::string> guide = std::nullopt) override;

private:
    LlgTokenizerPtr mTokenizer;
};
} // namespace tensorrt_llm::batch_manager
