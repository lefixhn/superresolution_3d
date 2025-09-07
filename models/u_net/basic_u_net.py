import torch.nn as nn 
import torch 
import torch.nn.functional as F 

# Data Structure: 
# 5D Tensor (BatchIndex, Channel, Depth, Height, Width)


class PixelShuffle3D(nn.Module):
    def __init__(self, upscale_factor):
        super().__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        batch_size, channels, depth, height, width = x.size()
        r = self.upscale_factor

        # Anzahl der Output-Kanäle
        out_channels = channels // (r ** 3)

        # Umformen: Kanäle in r³ aufteilen
        x = x.view(batch_size, out_channels, r, r, r, depth, height, width)

        # Umsortieren: (B, C, r, r, r, D, H, W) → (B, C, D*r, H*r, W*r)
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        x = x.view(batch_size, out_channels, depth * r, height * r, width * r)

        return x


class DenseBlock3D(nn.Module): 

    def __init__(self, in_channels, out_channels=None,num_layers=4, l_lrelu_alpha=0.1): 
        super().__init__()
        if out_channels = None: 
            out_channels = in_channels
        
        self.in_channels = in_channels
        self.out_channels = out_channels
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

    def __init__(self, upscale_factor=2):
        super().__init__()
        self.upscale_factor = upscale_factor
        entry_channels = 128

        level1_out_channels = 128
        level2_out_channels = 256
        level3_out_channels = 512

        self.encoder_level1 = nn.Sequential([
            nn.Conv3d(in_channels=1, out_channels=level1_out_channels), 
            nn.LeakyReLU(0.1),
            nn.Conv3d(in_channels=level1_out_channels, out_channels=level1_out_channels), 
            nn.LeakyReLU(0.1),
            nn.Conv3d(in_channels=level1_out_channels, out_channels=level1_out_channels), 
            nn.LeakyReLU(0.1),
            nn.Conv3d(in_channels=level1_out_channels, out_channels=level1_out_channels), 
            nn.LeakyReLU(0.1),
        ])

        self.encoder_level2 = nn.Sequential([
            nn.Conv3d(in_channels=level1_out_channels, out_channels=level2_out_channels), 
            nn.LeakyReLU(0.1),
            nn.Conv3d(in_channels=level2_out_channels, out_channels=level2_out_channels), 
            nn.LeakyReLU(0.1),
            nn.Conv3d(in_channels=level2_out_channels, out_channels=level2_out_channels), 
            nn.LeakyReLU(0.1),
            nn.Conv3d(in_channels=level2_out_channels, out_channels=level2_out_channels), 
            nn.LeakyReLU(0.1),
        ])

        self.encoder_level3 = nn.Sequential([
            nn.Conv3d(in_channels=level2_out_channels, out_channels=level3_out_channels), 
            nn.LeakyReLU(0.1),
            nn.Conv3d(in_channels=level3_out_channels, out_channels=level3_out_channels), 
            nn.LeakyReLU(0.1),
            nn.Conv3d(in_channels=level3_out_channels, out_channels=level3_out_channels), 
            nn.LeakyReLU(0.1),
            nn.Conv3d(in_channels=level3_out_channels, out_channels=level3_out_channels), 
            nn.LeakyReLU(0.1),
        ])

        decoder_level2_in_channels = level3_out_channels + level2_out_channels
        decoder_level2_out_channels = level2_out_channels

        decoder_level1_in_channels = decoder_level2_out_channels + level1_out_channels
        decoder_level1_out_channels = level1_out_channels
        

       
        self.decoder_level2 = nn.Sequential([
            nn.Conv3d(in_channels=decoder_level2_in_channels, out_channels=decoder_level2_out_channels), 
            nn.LeakyReLU(0.1),
            nn.Conv3d(in_channels=decoder_level2_out_channels, out_channels=decoder_level2_out_channels), 
            nn.LeakyReLU(0.1),
            nn.Conv3d(in_channels=decoder_level2_out_channels, out_channels=decoder_level2_out_channels), 
            nn.LeakyReLU(0.1),
            nn.Conv3d(in_channels=decoder_level2_out_channels, out_channels=decoder_level2_out_channels), 
            nn.LeakyReLU(0.1),
        ])

        self.decoder_level1 = nn.Sequential([
            nn.Conv3d(in_channels=decoder_level1_in_channels, out_channels=decoder_level1_out_channels), 
            nn.LeakyReLU(0.1),
            nn.Conv3d(in_channels=decoder_level1_out_channels, out_channels=decoder_level1_out_channels), 
            nn.LeakyReLU(0.1),
            nn.Conv3d(in_channels=decoder_level1_out_channels, out_channels=decoder_level1_out_channels), 
            nn.LeakyReLU(0.1),
            nn.Conv3d(in_channels=decoder_level1_out_channels, out_channels=decoder_level1_out_channels), 
            nn.LeakyReLU(0.1)
        ])
        
    
    def forward(self, x):
        
        encoder_level1_out = self.encoder_level1(x)
        
        encoder_level2_out = self.encoder_level2(encoder_level1_out)
        encoder_level3_out = self.encoder_level2(encoder_level2_out)
        


        

        

