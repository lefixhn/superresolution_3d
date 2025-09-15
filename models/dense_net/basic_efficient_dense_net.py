import torch
import torch.nn as nn 

class DenseUnit(nn.Module): 
    def __init__(self, in_channels, bottleneck_channels=48 ,growth_rate=12, build_activation_function=None, with_batch_norm=True, pre_activation=True): 
        super().__init__()
        self.in_channels = in_channels
        self.bottleneck_channels = bottleneck_channels
        self.growth_rate = growth_rate
        self.build_activation_function = build_activation_function
        self.with_batch_norm = with_batch_norm

        self.bottleneck_batch_norm = nn.BatchNorm3d(in_channels)
        self.bottleneck_activation = nn.LeakyReLU(0.1) if build_activation_function is None else build_activation_function()
        self.bottleneck_convolution = nn.Conv3d(
            in_channels = in_channels,
            out_channels = bottleneck_channels, 
            kernel_size = 1, 
            padding = 0
        )

        self.extraction_batch_norm = nn.BatchNorm3d(bottleneck_channels)
        self.extraction_activation = nn.LeakyReLU(0.1) if build_activation_function is None else build_activation_function()
        self.extraction_convolution = nn.Conv3d(
            in_channels = bottleneck_channels, 
            out_channels = growth_rate, 
            kernel_size = 3, 
            padding = 1
        )
    def forward(self, x): 
        # BOTTLENECK LAYER 
        if self.with_batch_norm: 
            x = self.bottleneck_batch_norm(x)
        # Control the order in wich the operations are being executed 
        if self.pre_activation: 
            x = self.bottleneck_convolution(self.bottleneck_activation(x))
        else: 
            x = self.bottleneck_activation(self.bottleneck_convolution(x))
        # EXTRACTION LAYER
        if self.with_batch_norm: 
            x = self.extraction_batch_norm(x)
        if self.pre_activation: 
            x = self.extraction_convolution(self.extraction_activation(x))
        else: 
            x = self.extraction_activation(self.extraction_convolution(x))
        return x


# Attention: the output here are only the newly extraxted features
class DenseBlock(nn.Module): 
    def __init__(self, in_channels, num_layers=8, growth_rate=12, bottleneck_channels=48, with_batch_norm=True, build_activation_function=None, pre_activation=True): 
        super().__init__()
        self.in_channels = in_channels
        self.num_layers = num_layers
        self.growth_rate = growth_rate
        self.bottleneck_channels = bottleneck_channels
        self.with_batch_norm = with_batch_norm
        self.build_activation_function = build_activation_function
        self.pre_activation = pre_activation

        self.dense_units = nn.ModuleList(
            [
                DenseUnit(
                    in_channels = in_channels + i*growth_rate, 
                    bottleneck_channels = bottleneck_channels, 
                    growth_rate = growth_rate, 
                    build_activation_function=build_activation_function, 
                    with_batch_norm=with_batch_norm, 
                    pre_activation=pre_activation
                )
            ]
            for i in range(self.num_layers)
        )
    def forward(self, x): 
        input_features=x
        output=self.dense_units[0](input_features)
        concatenated_outputs=output
        for i in range(1, self.num_layers): 
            output = self.dense_units[i](torch.cat([input_features, concatenated_outputs], dim=1))
            concatenated_outputs = torch.cat([concatenated_outputs, output], dim=1)
        return concatenated_outputs




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


