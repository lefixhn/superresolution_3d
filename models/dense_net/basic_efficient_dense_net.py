import torch
import torch.nn as nn 
# PixelShuffle3D was AI generated, because there is no implementation of it 
# in pytorch 
class PixelShuffle3D(nn.Module):
    """
    Rearranges elements in a tensor of shape (N, C*r^3, D, H, W)
    to a tensor of shape (N, C, D*r, H*r, W*r).
    """
    def __init__(self, upscale_factor: int):
        super().__init__()
        if upscale_factor < 1:
            raise ValueError("upscale_factor must be >= 1")
        self.r = upscale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            raise RuntimeError(f"PixelShuffle3D expects 5D input (N,C,D,H,W), got {x.dim()}D")
        n, c, d, h, w = x.shape
        r = self.r
        if c % (r ** 3) != 0:
            raise RuntimeError(f"Channel dim {c} not divisible by r^3={r**3}")
        c_out = c // (r ** 3)
        # (N, C_out*r^3, D, H, W) -> (N, C_out, D, r, H, r, W, r)
        x = x.view(n, c_out, r, r, r, d, h, w)
        # Permute to move r-dims next to their spatial dims
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4)  # (N, C_out, D, r, H, r, W, r)
        # Merge interleaved dims
        x = x.reshape(n, c_out, d * r, h * r, w * r)
        return x


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
    def __init__(self, in_channels, num_units=8, growth_rate=12, bottleneck_channels=48, with_batch_norm=True, build_activation_function=None, pre_activation=True): 
        super().__init__()
        self.in_channels = in_channels
        self.num_units = num_units
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
            for i in range(self.num_units)
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
    def __init__(
        self, 
        upscale_factor=2, 
        num_dense_blocks=8, 
        num_units_per_dense_block=8, 
        growth_rate=12, 
        bottleneck_channels=48, 
        with_batch_norm=True, 
        build_activation_function=None, 
        pre_activation=True
    ): 
        self.num_dense_blocks = num_dense_blocks
        self.num_units_per_dense_block = num_units_per_dense_block
        self.growth_rate = growth_rate
        self.bottleneck_channels = bottleneck_channels
        self.with_batch_norm = with_batch_norm
        # Set default value
        if build_activation_function is None: 
            build_activation_function = lambda: nn.LeakyReLU(0.1)
        self.build_activation_function = build_activation_function
        self.pre_activation = pre_activation
        # Calculated variables 
        self.dense_block_out_channels = growth_rate * num_units_per_dense_block
        self.dense_block_in_channels = 2 * growth_rate

        # One channel will be filled with the original image
        entry_out_chanels = bottleneck_channels
        self.entry = nn.Conv3d(in_channels=1, out_channels=entry_out_chanels-1)
        


        self.dense_blocks = nn.ModuleList([
            nn.Sequential([
                DenseBlock(
                in_channels = 2 * growth_rate, 
                num_units=num_units_per_dense_block, 
                growth_rate=growth_rate, 
                bottleneck_channels=bottleneck_channels, 
                with_batch_norm=with_batch_norm, 
                build_activation_function=build_activation_function, 
                pre_activation=pre_activation
                )
            ])
            for i in range(num_dense_blocks)
        ])

        self.compressors = nn.ModuleList([
            nn.Sequential([

                build_activation_function(), 
                nn.Conv3d(
                    in_channels=dense_block_out_channels*i, 
                    out_channels=dense_block_in_channels, 
                    kernel_size=1, 
                    padding=0
                )
            ])
            for i in range(1, num_dense_blocks) 
        ])

        # 3x3x3 > 1x1x1 > pixelshuffle 
        upsmpling_in_channels = entry_out_chanels + dense_block_out_channels * num_dense_blocks
        self.upsampling = nn.Sequential([
            build_activation_function(), 
            nn.Conv3d(
                in_channels=upsmpling_in_channels, 
                out_channels=upsmpling_in_channels, 
                kernel_size=3, 
                padding=1
            ),
            build_activation_function(), 
            nn.Conv3d(in_channels=upsmpling_in_channels, out_channels=upscale_factor**3)

            PixelShuffle3D(upscale_factor=upscale_factor)
        ])

    def forward(self, x): 
        entry_out = self.entry(x)
        dense_block_output = self.dense_blocks[0](entry_out)
        dense_blocks_outputs_concatenated = dense_block_output
        for i in range(1, self.num_dense_blocks):
            compressor_input = torch.cat([entry_out, dense_blocks_outputs_concatenated], dim=1)
            dense_block_input = self.compressors[i-1](compressor_input)
            dense_block_output = self.dense_blocks[i](dense_block_input)
            dense_blocks_outputs_concatenated = torch.cat([dense_blocks_outputs_concatenated, dense_block_output], dim=1)
        
        upscaled = self.upsampling(torch.cat([entry_out ,dense_blocks_outputs_concatenated]))
        return upscaled 

if __name__ == "__main__": 
    print("Checking weather model works")
    # Check weather this works
    model = BasicEfficientDenseNet()
    x = torch.randn(2, 1, 64, 64, 64)  # [B,C,D,H,W], D/H/W % 4 == 0
    y = model(x)
    print("in :", x.shape)  # torch.Size([2, 1, 64, 64, 64])
    print("out:", y.shape)  # Erwartet: [2, 1, 128, 128, 128]
    assert y.shape == (2, 1, 128, 128, 128)


