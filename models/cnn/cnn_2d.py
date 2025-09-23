import torch 
import torch.nn as nn



class CNN3D(nn.Module): 

    def __init__(self, num_layers=5, num_channels=64, upscale_factor=2):
        super().__init__()
        assert num_layers >= 2, "At least two layers are required"
        self.num_layers = num_layers
        self.num_channels = num_channels
        self.upscale_factor = upscale_factor

        self.entry = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.num_channels, kernel_size=3, padding=1), 
            nn.LeakyReLU(0.1)
        )

        self.main_convs = nn.ModuleList([

            nn.Sequential(
                
                nn.Conv2d(in_channels=self.num_channels, out_channels=self.num_channels, kernel_size=3, padding=1), 
                nn.LeakyReLU(0.1),
            )
            for i in range(self.num_layers-2)
        ])

        self.upscale = nn.Sequential(
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=self.num_channels, out_channels=upscale_factor**3, kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor=self.upscale_factor)
        )

    
    def forward(self, x): 
        x = self.entry(x)
        for conv in self.main_convs: 
            x = conv(x)
        x = self.upscale(x)
        return x

# Check weather it can handle a brats image
if __name__ == "__main__": 
    side_length = 64
    num_channels=512
    model = CNN3D(num_channels=num_channels)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).train()
    x = torch.randn(1, 1, side_length, side_length).to(device)
    
    y = model(x)
    assert y.shape == (1, 1, 2*side_length, 2*side_length)
    print(f"CNN3D Finished sl:{side_length} nc: {num_channels}")