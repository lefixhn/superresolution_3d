import torch.nn as nn 
import torch 
import torch.nn.functional as F 

class UNet(nn.Module): 
    def __init__(self, upscale_factor): 
        super().__init__()
        self.upscale_factor = upscale_factor

    
    def forward(self, x):
        return x 
    
    
    def _build_encoder_block(self, in_channels, out_channels):
        return None

    
    def _build_decoder_block(self, in_channels, out_channels):
        return None

    def _build_ds_method(self):
        return None 

    def _build_us_method(self): 
        return None

