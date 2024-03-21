#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/ngramPenalty.h"

using namespace tensorrt_llm::common;
using namespace tensorrt_llm::runtime;

namespace tensorrt_llm
{
namespace kernels
{

__global__ void calc_ngram_penalty(int32_t* workspace,
                                   SizeType const* inputLengths,
                                   SizeType const* sequenceLengths,
                                   TokenIdType const** outputIdsPtr,
                                   FinishedState const* finished,
                                   SizeType const* batchSlots,
                                   SizeType vocabSizePadded)
{
    SizeType constexpr maxNgramSize = 4;
    SizeType constexpr numNgrams = 4;
    SizeType constexpr ngramSizes[] = {1, 2, 3, 4};

    auto const batchIdx = static_cast<SizeType>(blockIdx.x);
    auto const batchSlot = batchSlots != nullptr ? batchSlots[batchIdx] : batchIdx;
    auto const inputLength = inputLengths[batchSlot];
    auto const sequenceLength = sequenceLengths[batchSlot];

    extern __shared__ TokenIdType sharedTokens[];
    auto const sharedTokensLength = static_cast<SizeType>(blockDim.x + maxNgramSize - 1);
    SizeType const lastTokensLength = maxNgramSize - 1;
    TokenIdType* lastTokens = &sharedTokens[sharedTokensLength];

    // if the beam has already finished, skip ngram check
    if ((finished != nullptr) && (finished[batchSlot].isFinished()))
    {
        return;
    }

    for (auto currTokenIdx = static_cast<SizeType>(inputLength + threadIdx.x); currTokenIdx < sequenceLength;
         currTokenIdx += static_cast<SizeType>(blockDim.x))
    {
        // write the tokens of the current block to shared memory
        sharedTokens[threadIdx.x] = outputIdsPtr[batchSlot][currTokenIdx];

        // write the next (maxNgramSize - 1) tokens after the current block, as well as the last (maxNgramSize - 1) tokens
        if (threadIdx.x == 0)
        {
            SizeType tokenIdx = currTokenIdx + static_cast<SizeType>(blockDim.x);
            SizeType bound = min(tokenIdx + sharedTokensLength, sequenceLength);
            for (auto i = static_cast<SizeType>(blockDim.x); tokenIdx < bound; ++i, ++tokenIdx)
            {
                sharedTokens[i] = outputIdsPtr[batchSlot][tokenIdx];
            }

            bound = max(sequenceLength - lastTokensLength, inputLength);
            for (SizeType i = 0, tokenIdx = sequenceLength - 1; tokenIdx >= bound; ++i, --tokenIdx)
            {
                lastTokens[lastTokensLength - 1 - i] = outputIdsPtr[batchSlot][tokenIdx];
            }
        }

        __syncthreads();

        #pragma unroll
        for (SizeType i = 0; i < numNgrams; ++i)
        {
            if (currTokenIdx > sequenceLength - ngramSizes[i])
            {
                continue;
            }

            bool ngramMatch = true;

            // in case of ngram size = 1 we will not run the for loop
            #pragma unroll
            for (SizeType ngramIdx = 0; ngramIdx < ngramSizes[i] - 1; ++ngramIdx)
            {
                if (sharedTokens[threadIdx.x + ngramIdx] != lastTokens[maxNgramSize - ngramSizes[i] + ngramIdx])
                {
                    ngramMatch = false;
                    break;
                }
            }

            if (ngramMatch)
            {
                SizeType tokenIdx = sharedTokens[threadIdx.x + ngramSizes[i] - 1];
                SizeType workspaceIdx = batchIdx * vocabSizePadded  + tokenIdx;
                int32_t bitPos = ngramSizes[i] - 1;
                atomicOr(&workspace[workspaceIdx], (int32_t)1 << bitPos);
            }
        }
    }
}

template <typename T>
__global__ void apply_ngram_penalty(T* logits,
                                    int32_t* workspace,
                                    SizeType const* batchSlots,
                                    float const* penalties,
                                    SizeType vocabSizePadded)
{
    SizeType constexpr numNgrams = 4;
    SizeType constexpr ngramSizes[] = {1, 2, 3, 4};

    auto const batchIdx = static_cast<SizeType>(blockIdx.x);
    auto const batchSlot = batchSlots != nullptr ? batchSlots[batchIdx] : batchIdx;
    float penalty = penalties[batchSlot];

    for (auto index = static_cast<SizeType>(threadIdx.x); index < vocabSizePadded;
         index += static_cast<SizeType>(blockDim.x))
    {
        #pragma unroll
        for (SizeType i = 0; i < numNgrams; ++i)
        {
            SizeType workspaceIdx = batchIdx * vocabSizePadded + index;
            int32_t bitPos = ngramSizes[i] - 1;
            if ((workspace[workspaceIdx] >> bitPos) & (int32_t)1)
            {
                auto logit = static_cast<float>(logits[batchIdx * vocabSizePadded + index]);
                logit = logit < 0.0f ? logit * penalty : logit / penalty;
                logits[batchIdx * vocabSizePadded + index] = logit;
            }
        }
    }
}

template <typename T>
void invokeNgramPenalty(T* logits,
                        int32_t* workspace,
                        SizeType const* inputLengths,
                        SizeType const* sequenceLengths,
                        TokenIdType const** outputIdsPtr,
                        FinishedState const* finished,
                        SizeType const* batchSlot,
                        float const* penalties,
                        SizeType batchSize,
                        SizeType vocabSizePadded,
                        cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(workspace, "no workspace provided for ngram penalty");

    constexpr SizeType maxNgramSize = 4;
    cudaMemsetAsync(workspace, 0, batchSize * vocabSizePadded * sizeof(int32_t), stream);
    
    dim3 block(256);
    dim3 grid(batchSize);

    // allocate shared memory of [blockDim + 2*(maxNgramSize - 1)] size, 
    // where 2*(maxNgramSize - 1) is for boundary token's ngram and for most recent generated tokens
    calc_ngram_penalty<<<grid, block, (block.x + 2 * (maxNgramSize - 1)) * sizeof(TokenIdType), stream>>>(
        workspace,
        inputLengths,
        sequenceLengths,
        outputIdsPtr,
        finished,
        batchSlot,
        vocabSizePadded);

    apply_ngram_penalty<<<grid, block, 0, stream>>>(
        logits,
        workspace,
        batchSlot,
        penalties,
        vocabSizePadded);
}

#define INVOKE_NGRAM_PENALTY(T)                                         \
    template void invokeNgramPenalty(T* logits,                         \
                                     int32_t* workspace,                \
                                     SizeType const* inputLengths,      \
                                     SizeType const* sequenceLengths,   \
                                     TokenIdType const** outputIdsPtr,  \
                                     FinishedState const* finished,     \
                                     SizeType const* batchSlot,         \
                                     float const* penalties,            \
                                     SizeType batchSize,                \
                                     SizeType vocabSizePadded,          \
                                     cudaStream_t stream);

INVOKE_NGRAM_PENALTY(float)
INVOKE_NGRAM_PENALTY(half)
#ifdef ENABLE_BF16
INVOKE_NGRAM_PENALTY(__nv_bfloat16)
#endif
#undef INVOKE_NGRAM_PENALTY

} // namespace kernels

} // namespace tensorrt_llm
