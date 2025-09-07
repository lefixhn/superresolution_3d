import torch.nn as nn 
import torch 
import torch.nn.functional as F 

# Data Structure: 
# 5D Tensor (BatchIndex, Channel, Depth, Height, Width)

class DenseBlock3D(nn.Module): 

    def __init__(self, in_channels ,num_layers=4, l_lrelu_alpha=0.1): 
        super().__init__()
        # The layers have growing input sizes due to concatination, but the
        # Same output size 
        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            nn.Sequential([
                nn.Conv3d(in_channels*i, in_channels, ),
                nn.LeakyReLU(l_lrelu_alpha)
            ])
            for i in range(1, num_layers+1)
        ])
    
    def forward(self, x): 
        for layer in self.layers: 
            x = torch.cat([x, layer(x)], dim=1)
        return x



class BasicUNetU(nn.Module): 

    def __init__(self, upscale_factor=2, num_blocks=4):
        super().__init__()
        self.upscale_factor = upscale_factor
        # 4 Blocks, after each one we scale down
        self.encoder_blocks = nn.ModuleList([        
                DenseBlock3D(64)
                for i in range(num_blocks)
        ])

        self.decoder_blocks = nn.ModuleList([
            DenseBlock3D(64*i)
            for i in range(1, self.num_blocks+1)
        ])
    
    def forward(self, x):
        # Stores the outputs of the encoder layers 
        encoder_results = []

        for block in self.encoder_blocks: 
            

        

        

