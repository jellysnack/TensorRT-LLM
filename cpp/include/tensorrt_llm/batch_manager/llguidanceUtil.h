#include "llguidance.h"
#include "tensorrt_llm/executor/executor.h"

using LlgMatcherPtr = std::shared_ptr<LlgMatcher>;
using LlgTokenizerPtr = std::shared_ptr<LlgTokenizer>;

LlgTokenizerPtr llgCreateTokenizer(tensorrt_llm::executor::GuidedDecodingConfig const& guidedDecodingConfig,
    tensorrt_llm::runtime::SizeType32 vocabSize);

bool llgValidateGrammar(tensorrt_llm::executor::GuidedDecodingParams const& guidedDecodingParams,
    LlgTokenizerPtr tokenizer, std::string* errorMessage);

std::string llgAddSchemaGuidance(std::string const& jsonStr);
