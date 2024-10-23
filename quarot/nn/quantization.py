import quarot
import torch

class Quantizer(torch.nn.Module):
    def __init__(self, input_clip_ratio=1.0):
        super().__init__()
        self.input_clip_ratio = input_clip_ratio
    
    def forward(self, x):
        scales_x = (torch.max(torch.abs(x), dim=-1)[0].unsqueeze(1)/7).to(torch.float16) * self.input_clip_ratio
        quantized_x = quarot.sym_quant(x, scales_x)
        packed_tensor = quarot.PackedQuantizedTensor(quantized_x, scales_x)
        return packed_tensor


class OurQuantizer(torch.nn.Module):
    def __init__(self, input_clip_ratio=1.0):
        super().__init__()
        self.input_clip_ratio = input_clip_ratio
    
    def forward(self, x):
        #print('before scale')
        scales_x = (torch.max(torch.abs(x), dim=-1)[0].unsqueeze(1)/7).to(torch.float16) * self.input_clip_ratio
        #print('after scale')
        #print(x.size(), scales_x.size())
        quantized_x = quarot.sym_quant_half(x, scales_x)
        #print(quantized_x.size(), quantized_x.dtype)
        packed_tensor = quarot.PackedNUQuantizedTensor(quantized_x, scales_x)
        return packed_tensor