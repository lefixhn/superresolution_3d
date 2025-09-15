import torch
import torch.nn as nn 


# Attention: the output here are only the newly extraxted features
class DenseBlock(nn.Module): 
    def __init__(self, in_channels , num_layers=8, growth_rate=12, bottleneck_channels=48, with_batch_norm=True, build_activation_function=None): 
        super().__init__()
        self.in_channels = in_channels
        self.num_layers = num_layers
        self.growth_rate = growth_rate
        self.bottleneck_channels = bottleneck_channels
        
        self.bottleneck_layers = nn.ModuleList([
            nn.Conv3d(
                in_channels = in_channels + i*growth_rate, 
                out_channels = bottleneck_channels, 
                kernel_size = 1, 
                padding = 0
            )
            for i in range(self.num_layers)
        ])

        self.extraction_layers = nn.ModuleList([
            nn.Sequential([
                nn.Conv3d(
                in_channels = bottleneck_channels, 
                out_channels = growth_rate, 
                kernel_size = 3, 
                padding = 1
            ), 
            # Batch Norm is optional
            if with_batch_norm: nn.BatchNorm3D(bottleneck_channels)
            # LReLU is default as activation function 
            if build_activation_function is None nn.LeakyReLU(0.1) else build_activation_function(), 
            ])
            for i in range(self.num_layers)
        ])

    def forward(self, x): 
        in_features = x
        out = None 
        for i in range(self.num_layers): 
            # Reduce features
            out = self.bottleneck_layers[i](in_features)
            # Extract
            out = self.extraction_layers[i](out)
            if i < self.num_layers - 1: 
                # Concatenate for the next layer
                in_features = torch.cat([in_features, out], dim=1)

        return out


class BasicEfficientDenseNet(nn.Module): 
    def __init__(self): 


    def forward(self): 

if __name__ == "__main__": 
    print("Checking weather model works")


    # Check weather this works
    model = BasicEfficientDenseNet()
    x = torch.randn(2, 1, 64, 64, 64)  # [B,C,D,H,W], D/H/W % 4 == 0
    y = model(x)
    print("in :", x.shape)  # torch.Size([2, 1, 64, 64, 64])
    print("out:", y.shape)  # Erwartet: [2, 1, 128, 128, 128]
    assert y.shape == (2, 1, 128, 128, 128)


