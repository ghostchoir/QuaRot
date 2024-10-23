// kernels.cu

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#define TILE_H 32
#define TILE_N 32
#define TILE_D 32

// First Kernel: Optimized Construction of the BF16 LUT

__global__ void construct_LUT_optimized(
    const __half * __restrict__ What,    // Input: [H, N, 4] BF16
    const __half * __restrict__ Delta,   // Input: [T] BF16
    const __half * __restrict__ Xmin,    // Input: [T] BF16
    __half * __restrict__ LUT,           // Output: [T, H, N, 4, 2] BF16
    int T,
    int H,
    int N
)
{
    // Calculate global indices
    int h0 = blockIdx.y * TILE_H;
    int n0 = blockIdx.x * TILE_N;

    int thread_h = threadIdx.y; // 0 to TILE_H - 1
    int thread_n = threadIdx.x; // 0 to TILE_N - 1

    int h = h0 + thread_h;
    int n = n0 + thread_n;

    // Each block processes one 't' value
    int t = blockIdx.z;

    if (t >= T || h >= H || n >= N)
        return;

    // Shared memory for What tile
    __shared__ __half What_tile[TILE_H][TILE_N][4]; // [H][N][D=4]

    // Load What into shared memory
    for (int d = 0; d < 4; ++d)
    {
        What_tile[thread_h][thread_n][d] = What[((h * N + n) * 4) + d];
    }

    // Synchronize to ensure What_tile is loaded
    __syncthreads();

    // Load Xmin[t] and Delta[t] into registers
    __half Xmin_t = Xmin[t];
    __half Delta_t = Delta[t];

    // Compute and store LUT values
    for (int d = 0; d < 4; ++d)
    {
        __half What_val = What_tile[thread_h][thread_n][d];

        // Compute LUT values using multiplication
        __half lut0 = __hmul(What_val, Xmin_t);
        __half lut1 = __hmul(What_val, Delta_t);

        // Compute the index in the LUT
        int lut_idx = ((((t * H + h) * N + n) * 4 + d) * 2);

        // Store the results
        LUT[lut_idx] = lut0;
        LUT[lut_idx + 1] = lut1;
    }
}

// Second Kernel: Optimized GEMM with Group-Quantized LLM Weights

__global__ void gemm_with_LUT_optimized(
    const int8_t * __restrict__ Wq,             // Input: [H, N, D/2] INT8
    const __half * __restrict__ LUT,     // Input: [T, H, N, 4, 2] BF16
    const __half * __restrict__ Xq,      // Input: [T, N, D] BF16
    __half * __restrict__ O,             // Output: [T, H] BF16
    int T,
    int H,
    int N,
    int D
)
{
    // Each block processes one (t, h) pair
    int t = blockIdx.x;
    int h = blockIdx.y;

    if (t >= T || h >= H)
        return;

    // Thread indices
    int n_thread = threadIdx.x; // 0 to TILE_N - 1
    int d_thread = threadIdx.y; // 0 to TILE_D - 1

    // Shared memory for Wq, Xq, and LUT
    __shared__ int8_t Wq_tile[TILE_N][TILE_D / 2];
    __shared__ __half Xq_tile[TILE_N][TILE_D];
    __shared__ __half LUT_tile[TILE_N][4][2];

    // Initialize partial sum
    __half partial_sum = __float2half(0.0f);
    // float partial_sum = 0.0f;

    // Loop over tiles of N and D
    for (int n_base = 0; n_base < N; n_base += TILE_N)
    {
        for (int d_base = 0; d_base < D; d_base += TILE_D)
        {
            int n = n_base + n_thread;
            int d = d_base + d_thread;

            // Load Wq into shared memory
            if (n < N && (d / 2) < (D / 2))
            {
                Wq_tile[n_thread][d_thread / 2] = Wq[((h * N + n) * (D / 2)) + (d / 2)];
            }

            // Load Xq into shared memory
            if (n < N && d < D)
            {
                Xq_tile[n_thread][d_thread] = Xq[((t * N + n) * D) + d];
            }

            // Load LUT into shared memory
            if (n < N && d_thread == 0)
            {
                // Each thread loads LUT for its 'n' and all 'wq_value's
                for (int wq_value = 0; wq_value < 4; ++wq_value)
                {
                    int lut_idx = ((((t * H + h) * N + n) * 4 + wq_value) * 2);
                    LUT_tile[n_thread][wq_value][0] = LUT[lut_idx];
                    LUT_tile[n_thread][wq_value][1] = LUT[lut_idx + 1];
                }
            }

            // Synchronize to ensure shared memory is loaded
            __syncthreads();

            // Compute partial sums
            if (n < N && d < D)
            {
                // Extract wq_value from Wq
                int8_t wq_byte = Wq_tile[n_thread][d_thread / 2];
                uint8_t wq_value;
                if (d % 2 == 0)
                    wq_value = wq_byte & 0x0F;       // Lower 4 bits
                else
                    wq_value = (wq_byte >> 4) & 0x0F; // Upper 4 bits

                wq_value &= 0x03; // Ensure wq_value is in [0, 3]

                // Fetch LUT values
                __half lut0 = LUT_tile[n_thread][wq_value][0];
                __half lut1 = LUT_tile[n_thread][wq_value][1];

                // Fetch Xq value
                __half Xq_val = Xq_tile[n_thread][d_thread];

                // Compute partial sum using fused multiply-add
                __half temp = __hfma(Xq_val, lut1, lut0);

                // Accumulate the sum
                partial_sum = __hadd(partial_sum, temp);

                // float lut0 = __half2float(LUT_tile[n_thread][wq_value][0]);
                // float lut1 = __half2float(LUT_tile[n_thread][wq_value][1]);

                // float Xq_val = __half2float( Xq_tile[n_thread][d_thread]);

                // float temp = __fma_rn(Xq_val, lut1, lut0);

                // partial_sum = __fadd_rn(partial_sum, temp);
            }

            // Synchronize before loading the next tile
            __syncthreads();
        }
    }

    // Perform intra-block reduction
    __shared__ __half shared_sum[TILE_N * TILE_D];

    int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
    shared_sum[thread_id] = partial_sum;
    //shared_sum[thread_id] = __float2half_rn(partial_sum);

    __syncthreads();

    // Reduction loop
    for (int stride = (blockDim.x * blockDim.y) / 2; stride > 0; stride >>= 1)
    {
        if (thread_id < stride)
        {
            shared_sum[thread_id] = __hadd(shared_sum[thread_id], shared_sum[thread_id + stride]);
        }
        __syncthreads();
    }

    // Thread 0 writes the result
    if (thread_id == 0)
    {
        O[t * H + h] = shared_sum[0];
    }
}

void construct_LUT_launcher(
    torch::Tensor What,    // [H, N, 4] BF16
    torch::Tensor Delta,   // [T] BF16
    torch::Tensor Xmin,    // [T] BF16
    torch::Tensor LUT,     // [T, H, N, 4, 2] BF16
    int T,
    int H,
    int N
)
{
    dim3 blockDim(TILE_N, TILE_H);
    dim3 gridDim(
        (N + TILE_N - 1) / TILE_N,
        (H + TILE_H - 1) / TILE_H,
        T
    );

    construct_LUT_optimized<<<gridDim, blockDim>>>(
        reinterpret_cast<__half*>(What.data_ptr()),
        reinterpret_cast<__half*>(Delta.data_ptr()),
        reinterpret_cast<__half*>(Xmin.data_ptr()),
        reinterpret_cast<__half*>(LUT.data_ptr()),
        T, H, N
    );

    //cudaDeviceSynchronize();
}

void gemm_with_LUT_launcher(
    torch::Tensor Wq,      // [H, N, D/2] INT8
    torch::Tensor LUT,     // [T, H, N, 4, 2] BF16
    torch::Tensor Xq,      // [T, N, D] BF16
    torch::Tensor O,       // [T, H] BF16
    int T,
    int H,
    int N,
    int D
)
{
    dim3 blockDim(TILE_N, TILE_D);
    dim3 gridDim(T, H);

    gemm_with_LUT_optimized<<<gridDim, blockDim>>>(
        Wq.data_ptr<int8_t>(),
        reinterpret_cast<__half*>(LUT.data_ptr()),
        reinterpret_cast<__half*>(Xq.data_ptr()),
        reinterpret_cast<__half*>(O.data_ptr()),
        T, H, N, D
    );
    //cudaDeviceSynchronize();
}

// // Wrapper functions for PyTorch

// void construct_LUT_launcher(
//     torch::Tensor What,    // [H, N, 4] BF16
//     torch::Tensor Delta,   // [T] BF16
//     torch::Tensor Xmin,    // [T] BF16
//     torch::Tensor LUT,     // [T, H, N, 4, 2] BF16
//     int T,
//     int H,
//     int N
// )
// {
//     dim3 blockDim(TILE_N, TILE_H);
//     dim3 gridDim(
//         (N + TILE_N - 1) / TILE_N,
//         (H + TILE_H - 1) / TILE_H,
//         T
//     );

//     construct_LUT_optimized<<<gridDim, blockDim>>>(
//         reinterpret_cast<__half*>(What.data_ptr()),
//         reinterpret_cast<__half*>(Delta.data_ptr()),
//         reinterpret_cast<__half*>(Xmin.data_ptr()),
//         reinterpret_cast<__half*>(LUT.data_ptr()),
//         T, H, N
//     );
// }

// void gemm_with_LUT_launcher(
//     torch::Tensor Wq,      // [H, N, D/2] INT8
//     torch::Tensor LUT,     // [T, H, N, 4, 2] BF16
//     torch::Tensor Xq,      // [T, N, D] BF16
//     torch::Tensor O,       // [T, H] BF16
//     int T,
//     int H,
//     int N,
//     int D
// )
// {
//     dim3 blockDim(TILE_N, TILE_D);
//     dim3 gridDim(T, H);

//     gemm_with_LUT_optimized<<<gridDim, blockDim>>>(
//         Wq.data_ptr<int8_t>(),
//         reinterpret_cast<__half*>(LUT.data_ptr()),
//         reinterpret_cast<__half*>(Xq.data_ptr()),
//         reinterpret_cast<__half*>(O.data_ptr()),
//         T, H, N, D
//     );
// }

// Binding code

// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//     m.def("construct_LUT", &construct_LUT_launcher, "Construct LUT (CUDA)");
//     m.def("gemm_with_LUT", &gemm_with_LUT_launcher, "GEMM with LUT (CUDA)");
// }
