import torch 
import torch.nn as nn

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

class CNN3D(nn.Module): 

    def __init__(self, num_layers=5, num_channels=64, upscale_factor=2):
        super().__init__()
        assert num_layers >= 2, "At least two layers are required"
        self.num_layers = num_layers
        self.num_channels = num_channels
        self.upscale_factor = upscale_factor

        self.entry = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=self.num_channels, kernel_size=3, padding=1), 
            nn.LeakyReLU(0.1)
        )

        self.main_convs = nn.ModuleList([

            nn.Sequential(
                
                nn.Conv3d(in_channels=self.num_channels, out_channels=self.num_channels, kernel_size=3, padding=1), 
                nn.LeakyReLU(0.1),
            )
            for i in range(self.num_layers-2)
        ])

        self.upscale = nn.Sequential(
            nn.LeakyReLU(0.1),
            nn.Conv3d(in_channels=self.num_channels, out_channels=upscale_factor**3, kernel_size=3, padding=1),
            PixelShuffle3D(upscale_factor=self.upscale_factor)
        )

    
    def forward(self, x): 
        x = self.entry(x)
        x = self.main_convs(x)
        x = self.upscale(x)
        return x

# Check weather it can handle a brats image
if __name__ == "__main__": 
    side_length = 128
    model = CNN3D(num_channels=512)
    x = torch.randn(1, 1, side_length, side_length, side_length)
    y = model(x)
    assert y.shape == (1, 1, 2*side_length, 2*side_length, 2*side_length)