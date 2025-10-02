import torch
import torch.nn as nn 
from torch.utils.checkpoint import checkpoint

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

class DenseUnit(nn.Module): 
    def __init__(self, in_channels, bottleneck_channels=48 ,growth_rate=12, build_activation_function=None, with_batch_norm=True, pre_activation=True): 
        super().__init__()
        self.in_channels = in_channels
        self.bottleneck_channels = bottleneck_channels
        self.growth_rate = growth_rate
        self.build_activation_function = build_activation_function
        self.with_batch_norm = with_batch_norm
        self.pre_activation = pre_activation

        self.bottleneck_batch_norm = nn.BatchNorm2d(in_channels)
        self.bottleneck_activation = nn.LeakyReLU(0.1) if build_activation_function is None else build_activation_function()
        self.bottleneck_convolution = nn.Conv2d(
            in_channels = in_channels,
            out_channels = bottleneck_channels, 
            kernel_size = 1, 
            padding = 0
        )

        self.extraction_batch_norm = nn.BatchNorm2d(bottleneck_channels)
        self.extraction_activation = nn.LeakyReLU(0.1) if build_activation_function is None else build_activation_function()
        self.extraction_convolution = nn.Conv2d(
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

        self.dense_units = nn.ModuleList([
            DenseUnit(
                in_channels = in_channels + i*growth_rate, 
                bottleneck_channels = bottleneck_channels, 
                growth_rate = growth_rate, 
                build_activation_function=build_activation_function, 
                with_batch_norm=with_batch_norm, 
                pre_activation=pre_activation
            )
            for i in range(self.num_units)
        ])
    def forward(self, x): 
        input_features=x
        concatenated_outputs=[]
        for i in range(self.num_units): 
            dense_unit_input = torch.cat([input_features] + concatenated_outputs, dim=1)
            output = self.dense_units[i](dense_unit_input)
            concatenated_outputs.append(output)
        return torch.cat(concatenated_outputs, dim=1)


class BasicEfficientDenseNet2d(nn.Module): 
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
        super().__init__()
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
        self.compressor_out_channels = 2 * growth_rate

        # One channel will be filled with the original image
        entry_out_chanels = 2 * growth_rate
        self.entry = nn.Conv2d(in_channels=1, 
            out_channels=entry_out_chanels-1, 
            kernel_size=3, 
            padding=1
        )

        self.dense_blocks = nn.ModuleList([
           
            DenseBlock(
                in_channels = entry_out_chanels + self.compressor_out_channels * i, 
                num_units=num_units_per_dense_block, 
                growth_rate=growth_rate, 
                bottleneck_channels=bottleneck_channels, 
                with_batch_norm=with_batch_norm, 
                build_activation_function=self.build_activation_function, 
                pre_activation=pre_activation
            )
            
            for i in range(num_dense_blocks)
        ])
        # A compressor after each denseblock 
        self.compressors = []
        for i in range(num_dense_blocks): 
            self.compressors.append(nn.Sequential(
                self.build_activation_function(), 
                nn.Conv2d(
                    in_channels=self.dense_block_out_channels, 
                    out_channels=self.compressor_out_channels, 
                    kernel_size=1, 
                    padding=0
                )
            ))   

        self.compressors = nn.ModuleList(self.compressors) 
        
        
        # Upsamoling gets all the features that have been extracted
        upsmpling_in_channels = self.compressor_out_channels * self.num_dense_blocks + entry_out_chanels
        self.upsampling = nn.Sequential(
            build_activation_function(), 
            nn.Conv2d(
                in_channels=upsmpling_in_channels, 
                out_channels=48, # Reduce to a reasonable amount
                kernel_size=3, 
                padding=1
            ),
            build_activation_function(), 
            nn.Conv2d(in_channels=48, out_channels=upscale_factor**2, kernel_size=1, padding=0),
            nn.PixelShuffle(upscale_factor=upscale_factor)
        )

    def forward(self, x): 
        entry_out = torch.cat([x, self.entry(x)], dim=1)
        aggregated_compressed_outputs = []
        input_features = entry_out
        
        for i in range(self.num_dense_blocks):
            if __name__ == "__main__": 
                print(f"BEFORE DENSE BLOCK {i}") # Debugging purpose 
            # Calculate denseblock output
            #dense_block_output =  checkpoint(self.dense_blocks[i],input_features, use_reentrant=False)
            dense_block_output =  self.dense_blocks[i](input_features)
            
            # Compress denseblock output
            compressed_dense_block_output = self.compressors[i](dense_block_output)
            # Add compressed denseblock output to list 
            aggregated_compressed_outputs.append(compressed_dense_block_output)
            # Calculate input_features for next iteration 
            input_features = torch.cat([entry_out] + aggregated_compressed_outputs, dim=1)
            
        upscaled = self.upsampling(input_features)
        return upscaled 




if __name__ == "__main__": 
    model = BasicEfficientDenseNet2d()
    lr_image = torch.randn(4, 1, 128, 128)
    hr_image = model(lr_image)
    if hr_image.shape == (4, 1, 256, 256):
        print("BasicEfficientDenseNet2d completed sucessfully")
    else: 
        raise ValueError(f"Wrong output shape {hr_image.shape}")
    
 