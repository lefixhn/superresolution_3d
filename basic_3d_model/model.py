import torch.nn as nn
import torch 
import torch.nn.functional as F

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

class FirstRes3DModel(nn.Module): 
    def __init__(self, upscale_factor=2):
        super().__init__()
        self.base_upscale = nn.ConvTranspose3d(1, 1, kernel_size=4, stride=2, padding=1)
        self.upscale_factor = upscale_factor
        self.conv1 = nn.Conv3d(1, 32, 3, 1, 1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv3d(32, 32, 3, 1, 1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv3d(32, 8, 3, 1, 1)
        self.pixelshuffle = PixelShuffle3D(upscale_factor=self.upscale_factor)

    def forward(self, x):
        # Upscaling base 
        # TODO: check weather mode trilinear is maybe stupid, because it is to blurry
        x_upscaled =  self.base_upscale(x)
        # Calculating residual
        residual = self.conv1(x)
        residual = self.relu1(residual)
        residual = self.conv2(residual)
        residual = self.relu2(residual)
        residual = self.conv3(residual)
        residual = self.pixelshuffle(residual)
        result = x_upscaled + residual
        return result

if __name__ == "__main__":
    # Try model out
    x  = torch.randn(1, 1, 25, 25, 25)
    model = FirstRes3DModel(upscale_factor=2)
    y = model(x)
