#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/ngramPenalty.h"

using namespace tensorrt_llm::common;

namespace tensorrt_llm
{
namespace kernels
{

template <typename T>
__global__ void ngram_penalty(T* logits, const int** output_ids_buf, const FinishedState* finished_buf,
    const int** parent_ids_buf, const int* batch_slots, int batch_size, int beam_width, int max_seq_len,
    int vocab_size_padded, const int* sequence_lengths)
{
    constexpr int max_ngram_size = 4;
    constexpr int num_ngram_sizes = 4;
    constexpr int ngram_sizes[] = {1, 2, 3, 4};
    const T penalties[] = {0.5, 1.0, 2.0, 4.0};

    const int output_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int local_batch_idx = blockIdx.y / beam_width;
    auto const batch_slot = batch_slots != nullptr ? batch_slots[local_batch_idx] : local_batch_idx;
    const int beam_idx = blockIdx.y % beam_width;
    const bool beam_search = beam_width > 1;
    const int step = sequence_lengths[batch_slot];

    // if the beam has already finished, skip ngram check
    if ((finished_buf != nullptr) && (finished_buf[batch_slot * beam_width + beam_idx].isFinished()))
    {
        return;
    }

    // shared mem layout: per-thread token that each thread is mapped to, plus (ngram_size - 1) extra tokens beyond
    // block boundary, plus (ngram_size - 1) most recent generated tokens in the beam
    extern __shared__ int shared_tokens[];
    int shared_tokens_length = blockDim.x + max_ngram_size - 1;
    int* last_tokens = &shared_tokens[shared_tokens_length];
    int last_tokens_length = max_ngram_size - 1;

    // retrieve the entire beam by backtracking from last token to current token  (in reverse order)
    // single thread vs parallel thread is equivalent as it's bound by the longest iteration
    if (threadIdx.x == 0)
    {
        int parent_id = beam_idx;
        int start_record_idx = min(output_idx + shared_tokens_length, (int) step);
        int shared_token_idx = start_record_idx == step ? step - output_idx - 1 : shared_tokens_length - 1;
        int last_token_idx = last_tokens_length - 1;
        // write to shared mem in reverse order; boundary condition when thread block covers more than step

        for (int curr_idx = step - 1; curr_idx >= output_idx; curr_idx--)
        {
            if (last_token_idx >= 0)
            {
                last_tokens[last_token_idx--] = output_ids_buf[batch_slot][parent_id * max_seq_len + curr_idx];
            }

            // before reaching the part of current block, traverse only; after that, record the tokens
            if (curr_idx < start_record_idx)
            {
                shared_tokens[shared_token_idx--] = output_ids_buf[batch_slot][parent_id * max_seq_len + curr_idx];
            }

            if (beam_search)
            {
                parent_id = parent_ids_buf[batch_slot][parent_id * max_seq_len + curr_idx];
            }
        }
    }

    __syncthreads();

    #pragma unroll
    for (int i = 0; i < num_ngram_sizes; ++i) {

        if (output_idx > step - ngram_sizes[i])
        {
            continue;
        }

        bool ban_ngram = true;

        // in case of ngram size = 1 we will not run the for loop
        #pragma unroll
        for (int ngram_idx = 0; ngram_idx < ngram_sizes[i] - 1; ngram_idx++)
        {
            if (shared_tokens[threadIdx.x + ngram_idx] != last_tokens[max_ngram_size - ngram_sizes[i] + ngram_idx])
            {
                ban_ngram = false;
                break;
            }
        }

        if (ban_ngram)
        {
            int banned_token = shared_tokens[threadIdx.x + ngram_sizes[i] - 1];
            atomicAdd(&logits[local_batch_idx * beam_width * vocab_size_padded + beam_idx * vocab_size_padded + banned_token], -penalties[i]);
        }
    }
}

template <typename T>
void invokeNgramPenalty(T* logits, const int** output_ids_buf, const FinishedState* finished_buf,
    const int** parent_ids_buf, const int* batch_slot, const int* sequence_lengths, int batch_size, int beam_width,
    int max_seq_len, int vocab_size_padded, size_t max_step, cudaStream_t stream)
{
    // each input in the local batch can have different no_repeat_ngram_size. Use max for shmem allocation
    // getting the max of current batch and allocate shmem as needed is ideal. But here the ngram_buf is on GPU, while
    // this max value is on CPU for kernel launch. Instead of really finding the max and extra CPU-GPU memcpy, we simply
    // use a constant. In practice, ngram size is usually very small, like 3 or 4.
    constexpr int max_ngram_size = 32;

    // step (current generated length, except start token) is from 1 ~ max_seq_len
    dim3 block, grid;
    constexpr size_t max_blocks{256};
    block.x = min(((max_step + 32 - 1) / 32) * 32, max_blocks);
    grid.x = (max_step + block.x - 1) / block.x;
    grid.y = batch_size * beam_width;

    // dynamically allocate shared memory of int[blockDim + 2*(ngram_size - 1)], where ngram_size - 1 is for boundary
    // token's ngram and for most recent tokens
    ngram_penalty<<<grid, block, (block.x + 2 * (max_ngram_size - 1)) * sizeof(int), stream>>>(logits,
        output_ids_buf, finished_buf, parent_ids_buf, batch_slot, batch_size, beam_width, max_seq_len,
        vocab_size_padded, sequence_lengths);
    sync_check_cuda_error();
}

#define INVOKE_NGRAM_PENALTY(T)                                                                                 \
    template void invokeNgramPenalty(T* logits, const int** output_ids_buf, const FinishedState* finished_buf,  \
        const int** parent_ids_buf, const int* batch_slot, const int* sequence_lengths, int batch_size,         \
        int beam_width, int max_seq_len, int vocab_size_padded, size_t max_step,                                \
        cudaStream_t stream);

INVOKE_NGRAM_PENALTY(float)
INVOKE_NGRAM_PENALTY(half)
#ifdef ENABLE_BF16
INVOKE_NGRAM_PENALTY(__nv_bfloat16)
#endif
#undef INVOKE_NGRAM_PENALTY

} // namespace kernels

} // namespace tensorrt_llm
