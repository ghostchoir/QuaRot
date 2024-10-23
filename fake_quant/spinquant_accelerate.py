import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
import torch
#print(torch.cuda.device_count())
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from random import randint

import tqdm

from rotation_utils import random_orthogonal_matrix
from hadamard_utils import random_hadamard_matrix, apply_exact_had_to_linear
from quant_utils import ActQuantWrapper

import utils
import model_utils
import data_utils
import transformers
import quant_utils
import rotation_utils
import gptq_utils
import eval_utils
import hadamard_utils

from transformers.trainer_pt_utils import LabelSmoother

import rotated_llama
from rotated_llama import RotatedLlamaForCausalLM

from stiefel import stiefel_optimizer


from accelerate import Accelerator
from datasets import Dataset
from torch.utils.data import DataLoader



def main():
    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    else:
        local_rank = 0
    args = utils.parser_gen()

    assert args.learn_r1 or args.learn_r2


    transformers.set_seed(args.seed)
    model = model_utils.get_model(args.model, args.hf_token)
    model.eval()

    trainloader = data_utils.get_loaders(
        args.cal_dataset, nsamples=args.nsamples,
        seed=args.seed, model=args.model,
        seqlen=model.seqlen, eval_mode=False
    )
    
    dataset = Dataset.from_dict({"data": [d.squeeze() for (d, _)  in trainloader]}).with_format("torch")
    trainloader = DataLoader(dataset, shuffle=True, batch_size=1, num_workers=2)


    with torch.no_grad():
        rotated_model = RotatedLlamaForCausalLM(model.config, model)
    rotated_model.eval()

    del model
    rotation_utils.fuse_layer_norms(rotated_model)
    for p in rotated_model.parameters():
        p.requires_grad_(False)


    quant_utils.add_actquant(
        rotated_model,
        layers=[nn.Linear,
                ActQuantWrapper,
                rotated_llama.RotatedLinear,
                rotated_llama.RotatedOVProj]
    )

    qlayers = quant_utils.find_qlayers(
        rotated_model.model,
        layers=[nn.Linear,
                ActQuantWrapper,
                rotated_llama.RotatedLinear,
                rotated_llama.RotatedOVProj]
    )

    for name in qlayers:
        if 'down_proj' in name:
            had_K, K = hadamard_utils.get_hadK(rotated_model.config.intermediate_size)
            hadamard_utils.apply_exact_had_to_linear(qlayers[name].module, had_dim=-1, output=False)
            qlayers[name].online_full_had = True
            qlayers[name].had_K = had_K
            qlayers[name].K = K
            qlayers[name].fp32_had = args.fp32_had

    if args.a_bits < 16 or args.v_bits < 16:
        qlayers = quant_utils.find_qlayers(rotated_model, layers=[quant_utils.ActQuantWrapper])
        down_proj_groupsize = -1
        if args.a_groupsize > 0 and "llama" in args.model:
            down_proj_groupsize = utils.llama_down_proj_groupsize(rotated_model, args.a_groupsize)
        
        for name in qlayers:            
            layer_input_bits = args.a_bits
            layer_groupsize = args.a_groupsize
            layer_a_sym = not(args.a_asym)
            layer_a_clip = args.a_clip_ratio
            
            if 'v_proj' in name and args.v_bits < 16: #Set the v_proj precision
                qlayers[name].out_quantizer.configure(bits=args.v_bits,
                                                groupsize=args.v_groupsize,
                                                sym=not(args.v_asym),
                                                clip_ratio=args.v_clip_ratio)
            
            if 'lm_head' in name: #Skip lm_head quantization   
                layer_input_bits = 16
            
            if 'down_proj' in name: #Set the down_proj precision
                if args.int8_down_proj:
                    layer_input_bits = 8
                layer_groupsize = down_proj_groupsize

                
            qlayers[name].quantizer.configure(bits=layer_input_bits,
                                                groupsize=layer_groupsize,
                                                sym=layer_a_sym,
                                                clip_ratio=layer_a_clip)

    if args.w_bits < 16:
        save_dict = {}
        if args.load_qmodel_path: # Load Quantized Rotated Model
            assert args.rotate, "Model should be rotated to load a quantized model!"
            assert not args.save_qmodel_path, "Cannot save a quantized model if it is already loaded!"
            print("Load quantized model from ", args.load_qmodel_path)
            save_dict = torch.load(args.load_qmodel_path)
            rotated_model.load_state_dict(save_dict["model"])
            
        elif not args.w_rtn: # GPTQ Weight Quantization
            assert "llama" in args.model, "Only llama is supported for GPTQ!"
            
            trainloader = data_utils.get_loaders(
                args.cal_dataset, nsamples=args.nsamples,
                seed=args.seed, model=args.model,
                seqlen=model.seqlen, eval_mode=False
            )
            quantizers = gptq_utils.gptq_fwrd(
                rotated_model,
                trainloader,
                utils.DEV,
                args,
                [rotated_llama.RotatedLinear, rotated_llama.RotatedOVProj])
            save_dict["w_quantizers"] = quantizers
        else: # RTN Weight Quantization
            quantizers = gptq_utils.rtn_fwrd(rotated_model, utils.DEV, args)
            save_dict["w_quantizers"] = quantizers
            
        if args.save_qmodel_path:
            save_dict["model"] = model.state_dict()
            torch.save(save_dict, args.save_qmodel_path)

    if args.k_bits < 16:
        if args.k_pre_rope:
            raise NotImplementedError("Pre-RoPE quantization is not supported yet!")
        else:
            rope_function_name = model_utils.get_rope_function_name(rotated_model)
            layers = model_utils.get_layers(rotated_model)
            k_quant_config = {'k_bits':args.k_bits, "k_groupsize": args.k_groupsize,
                                            "k_sym": not(args.k_asym), "k_clip_ratio": args.k_clip_ratio}
            for layer in layers:
                rotation_utils.add_qk_rotation_wrapper_after_function_call_in_forward(
                            layer.self_attn, 
                            rope_function_name, 
                            config=rotated_model.config,
                            **k_quant_config)

    for p in rotated_model.parameters():
        p.requires_grad_(False)
    #utils.distribute_model(rotated_model)

    params = []
    if args.learn_r1:
        #r1 = random_hadamard_matrix(rotated_model.config.hidden_size, utils.DEV).to(dtype=torch.float64)
        r1 = torch.load('R1.pt').to(torch.float16)
        R1 = nn.Parameter(r1, requires_grad=True)
        params.append(R1)
    else:
        R1 = None

    if args.learn_r2:
        R2s = []
        r2s = torch.load('R2s.pt')
        for i in range(rotated_model.config.num_hidden_layers):
            r2 = r2s[i].to(torch.float16)
            R2 = nn.Parameter(r2, requires_grad=True)
            R2s.append(R2)
            params.append(R2)
    else:
        R2s = None

    optimizer = stiefel_optimizer.SGDG([
        {'params': params, 'lr': 1.5, 'momentum': args.momentum, 'stiefel': True}
    ])
    def lr_schedule(iter, total_iter, max_lr, min_lr):
        return max_lr - iter / total_iter * (max_lr - min_lr)

    accelerator = Accelerator()

    optimizer, rotated_model, trainloader = accelerator.prepare(
        optimizer, rotated_model, trainloader
    )

    label_smoother = LabelSmoother(0.0)

    #idx_stack = None

    #pbar = tqdm.tqdm(range(0 + 1, 100 + 1), desc="Training progress", dynamic_ncols=True)
    #for iteration in pbar:
    for iteration, data in enumerate(trainloader):
        
        input = data['data']
        
        #input = data[0]#.to(utils.DEV)
        #target = data[1]#.to(utils.DEV)
        #print()
        output = rotated_model(input, R1=R1, R2s=R2s)
        print('forward')
        loss = label_smoother(output, input, shift_labels=True)
        
        optimizer.zero_grad()
        #loss.backward()
        print('backward')
        accelerator.backward(loss)
        lr = lr_schedule(iteration, 100, 1.5, 0)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        #print([p.grad is None for p in params])
        optimizer.step()
        print('step')
        if local_rank == 0:
            print(f'CE: {loss.item():.3f}')
        
        #with torch.no_grad():
        #    pbar.set_postfix(
        #        {'CE': f'{loss.item():.3f}',
        #        'ortho1': f'{(torch.matmul(R1, R1.T) - torch.eye(R1.size(0)).to(R1.device)).sum().item():.3f}',
        #        #'ortho2': f'{(torch.matmul(Q2, Q2.T) - torch.eye(Q2.size(0)).to(Q2.device)).sum().item():.3f}',
        #        #'det(Q)': f'{torch.linalg.det(Q):.3f}'
        #        }
        #    )

    print('Training complete. Saving learned rotation(s).')

    accelerator.wait_for_everyone()
    if local_rank == 0:
        if not os.path.exists('rotations'):
            os.makedirs('rotations', exist_ok=True)

        prefix = args.prefix_r + '_' if len(args.prefix_r) > 0 else ''

        if args.learn_r1:
            torch.save(R1.detach().cpu(), f'rotations/{prefix}R1.pt')
        if args.learn_r2:
            for i, R2 in enumerate(R2s):
                torch.save(R2.detach().cpu(), f'rotations/{prefix}R2_{i}.pt')


if __name__ == "__main__":
    main()