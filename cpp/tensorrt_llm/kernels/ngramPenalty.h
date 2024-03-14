#pragma once

#include "tensorrt_llm/kernels/decodingCommon.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace tensorrt_llm
{
namespace kernels
{

template <typename T>
void invokeNgramPenalty(T* logits, const int** output_ids_buf, const FinishedState* finished_buf,
    const int** parent_ids_buf, const int* batch_slot, const int* sequence_lengths, int batch_size, int beam_width,
    int max_seq_len, int vocab_size_padded, size_t max_step, cudaStream_t stream);

} // namespace kernels
} // namespace tensorrt_llm
