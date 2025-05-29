#include "tensorrt_llm/executor/executor.h"
#include "llguidance.h"

using LlgMatcherPtr = std::shared_ptr<LlgMatcher>;
using LlgTokenizerPtr = std::shared_ptr<LlgTokenizer>;

LlgTokenizerPtr llgCreateTokenizer(tensorrt_llm::executor::GuidedDecodingConfig const& guidedDecodingConfig,
                                    tensorrt_llm::runtime::SizeType32 vocabSize);

bool llgValidateGrammar(const tensorrt_llm::executor::GuidedDecodingParams& guidedDecodingParams,
                        LlgTokenizerPtr tokenizer,
                        std::string* errorMessage);

std::string llgAddSchemaGuidance(const std::string& jsonStr);
