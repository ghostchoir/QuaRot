import torch
from quarot.nn import Linear2bit, OurQuantizer, OnlineHadamard
import time
import argparse
import numpy as np
import pprint

model_sizes = [
    (4096, 4096), #llama-7b
    #(5120, 5120), #llama-13b
    #(8192, 8192)  #llama-70b   
]

mlp_sizes = [
    (4096, 11008), #llama-7b
    #(5120, 13824), #llama-13b
    #(8192, 28672)  #llama-70b
]
benchmark_dtypes = [torch.float16]
num_warmup_steps = 5
num_bench_steps = 10


def module_benchmark(module, x, sync=True):
    x = x.cuda()
    
    # warmup
    for i in range(num_warmup_steps):
        #print(i, x.size())
        out = module(x)
    if sync:
        torch.cuda.synchronize()
    
    start_time = time.perf_counter()
    for i in range(num_bench_steps):
        out = module(x)
    if sync:
        torch.cuda.synchronize()
    
    end_time = time.perf_counter()
    
    
    return (end_time - start_time) * 1000 / num_bench_steps

def linear2bit_benchmark(args):
        
    bsz = args.bsz
    seq_len = args.seq_len
    
    if args.layer_type == 'v_proj':
        layer_size = model_sizes
    else:
        layer_size = mlp_sizes
        
    
    for (feature_dim_in, feature_dim_out) in layer_size:
        for dtype in benchmark_dtypes:
            
            x = torch.rand((bsz * seq_len,
                            feature_dim_in)).cuda().to(dtype)
            
            baseline_mod = torch.nn.Linear(feature_dim_in,  #W
                                           feature_dim_out, #H
                                           bias=False).cuda().to(dtype)
            
            baseline_mod.weight.data = torch.randint_like(baseline_mod.weight.data,
                                                          low=-128, high=127).to(dtype)
            
            LUT = torch.zeros((bsz * seq_len, feature_dim_out, feature_dim_in // 128, 4, 2), dtype=torch.float16, requires_grad=False).cuda()
            
            O = torch.zeros((bsz * seq_len, feature_dim_out), dtype=torch.float16, requires_grad=False).cuda()
            #s_w = torch.ones((feature_dim_out, 1), dtype=torch.float16, device='cuda')
            int4_mod = torch.nn.Sequential(
                OurQuantizer(input_clip_ratio=1.0),
                Linear2bit.from_float(baseline_mod, group_size=128, LUT=LUT, O=O)
            ).cuda()
            int4_mod_had = torch.nn.Sequential(
                OnlineHadamard(baseline_mod.in_features, force_fp32=True),
                OurQuantizer(input_clip_ratio=1.0),
                Linear2bit.from_float(baseline_mod, group_size=128, LUT=LUT, O=O),
            ).cuda()
            #int4_mod_had.online_full_had = True
            #int4_mod.fp32_had = True
            
            int4_mod_fp16had = torch.nn.Sequential(
                OnlineHadamard(baseline_mod.in_features, force_fp32=False),
                OurQuantizer(input_clip_ratio=1.0),
                Linear2bit.from_float(baseline_mod, group_size=128, LUT=LUT, O=O),
            ).cuda()



            print(f"{dtype}. Sizes: {baseline_mod.weight.shape}")
            
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            times_baseline = []
            
            # for i in range(5):
            #     out = baseline_mod(x)
            # torch.cuda.synchronize()
            
            
            for i in range(10):
                start_time = time.perf_counter()
                out = baseline_mod(x)
                end.record()
                torch.cuda.synchronize()
                end_time = time.perf_counter()
                print((end_time - start_time))
            #for i in range(10):
            #    times_baseline.append(module_benchmark(baseline_mod, x))
            #print(f"FP16 time: {np.mean(times_baseline):.3f} +- {1.96 * np.std(times_baseline):.3f}ms")
            
            
            times_4bit = []
            
            
            
            for i in range(5):
                out = int4_mod(x)
            torch.cuda.synchronize()
            
            
            for i in range(10):
                start.record()
                out = int4_mod(x)
                end.record()
                torch.cuda.synchronize()
                op_time = start.elapsed_time(end)
                print(op_time / 1000)
            #torch.cuda.synchronize()
            
            #for i in range(10):
            #    times_4bit.append(module_benchmark(int4_mod, x, True))
            #print(f"Int4 time: {np.mean(times_4bit):.3f} +- {1.96 * np.std(times_4bit):.3f}ms")
            
            times_4bit_had = []
            for i in range(10):
                times_4bit_had.append(module_benchmark(int4_mod_had, x, True))
            print(f"Int4 (+FP32had) time: {np.mean(times_4bit_had):.3f} +- {1.96 * np.std(times_4bit_had):.3f}ms")
            
            times_4bit_fp16had = []
            for i in range(10):
                times_4bit_fp16had.append(module_benchmark(int4_mod_fp16had, x, True))
            print(f"Int4 (+FP16had) time: {np.mean(times_4bit_fp16had):.3f} +- {1.96 * np.std(times_4bit_fp16had):.3f}ms")
            
            
            
            print(f"Speedup: {np.mean(times_baseline) / np.mean(times_4bit):.3f}x")
            
            # table-style output
            print(f'{feature_dim_in}x{feature_dim_out} & {args.bsz} & {np.mean(times_baseline):.3f} & {np.mean(times_4bit):.3f} & {np.mean(times_4bit_had):.3f} & {np.mean(times_4bit_fp16had):.3f}\\\\')
            print('--------------')
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--bsz', type=int,
        help='Batch size',
        default=1,
    )
    parser.add_argument(
        '--seq_len', type=int,
        help='Size of the input sequence',
        default=2048,
    )
    parser.add_argument(
        '--layer_type', type=str,
        help='Type of the layer in the model (v_proj [default], down_proj)',
        default='v_proj',
        choices=['v_proj', 'down_proj']
    )
    
    args = parser.parse_args()
    pprint.pprint(vars(args))
    linear2bit_benchmark(args)
