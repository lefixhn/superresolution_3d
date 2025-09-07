import torch.nn as nn 
import torch 

# Data Structure: 
# 5D Tensor (BatchIndex, Channel, Depth, Height, Width)

class DenseBlock3D(nn.Module): 

    def __init__(self, in_channels ,layers=4 ): 
        super().__init__()
        # The layers have growing input sizes due to concatination, but the
        # Same output size 
        self.layers = [
            nn.Conv3d(in_channels*i, in_channels)
            for i in range(1, layers+1)
        ]
        

    
    def forward(self, x): 


class BasicUNetU(nn.Module): 

    def __init__(self, upscale_factor):
        super().__init__()
        self.upscale_factor = upscale_factor

    
    def forward(self, x):

