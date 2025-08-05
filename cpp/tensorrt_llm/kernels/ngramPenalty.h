#pragma once

#include "tensorrt_llm/kernels/decodingCommon.h"
#include "tensorrt_llm/runtime/common.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace tensorrt_llm
{
namespace kernels
{

void invokeNgramPenalty(int32_t* workspace, runtime::SizeType32 const* inputLengths,
    runtime::SizeType32 const* sequenceLengths, runtime::TokenIdType const** outputIdsPtr,
    FinishedState const* finished, runtime::SizeType32 const* batchSlot, runtime::SizeType32 batchSize,
    runtime::SizeType32 vocabSize, cudaStream_t stream);

} // namespace kernels
} // namespace tensorrt_llm
