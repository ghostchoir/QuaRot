#pragma once
#include <cuda_fp16.h>

void construct_LUT_launcher(
    torch::Tensor What,    // [H, N, 4] BF16
    torch::Tensor Delta,   // [T] BF16
    torch::Tensor Xmin,    // [T] BF16
    torch::Tensor LUT,     // [T, H, N, 4, 2] BF16
    int T,
    int H,
    int N
);

void gemm_with_LUT_launcher(
    torch::Tensor Wq,      // [H, N, D/2] INT8
    torch::Tensor LUT,     // [T, H, N, 4, 2] BF16
    torch::Tensor Xq,      // [T, N, D] BF16
    torch::Tensor O,       // [T, H] BF16
    int T,
    int H,
    int N,
    int D
);