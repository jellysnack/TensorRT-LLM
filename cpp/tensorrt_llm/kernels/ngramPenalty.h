#pragma once

#include "tensorrt_llm/kernels/decodingCommon.h"
#include "tensorrt_llm/runtime/common.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace tensorrt_llm
{
namespace kernels
{

template <typename T>
void invokeNgramPenalty(T* logits,
                        int32_t* workspace,
                        runtime::SizeType const* inputLengths,
                        runtime::SizeType const* sequenceLengths,
                        runtime::TokenIdType const** outputIdsPtr,
                        FinishedState const* finished,
                        runtime::SizeType const* batchSlot,
                        float const* penalties,
                        runtime::SizeType batchSize,
                        runtime::SizeType vocabSizePadded,
                        cudaStream_t stream);

} // namespace kernels
} // namespace tensorrt_llm
