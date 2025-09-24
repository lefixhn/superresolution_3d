'''
    Uses a 2D SR Model to generate a 3D HR Volume
    Two axes are upscaled though the 2D Model. 
    The remaining axe is upscaled though interpolation
'''
import torch 
from scipy import ndimage as nd
@torch.no_grad()
def build_sclie_wise_hr_volume(
    lr_volume: torch.Tensor, 
    model: callable, 
    interpolation_order=1, 
    interpolation_dim="D",     # Can be "D", "H", "W"
    upscale_factor=2, 
    device = "cuda" if torch.cuda.is_available() else "cpu", 
): 
    if lr_volume.ndim() == 4: 
        lr_volume = lr_volume.unsqueeze(0)
    elif lr_volume.ndim() == 3: 
        lr_volume = lr_volume.unsqueeze(0).unsqueeze(0)
    elif lr_volume.ndim() != 5: 
        raise TypeError("Input must have shape (B, C, D, H, W), (C, D, H, W) or (D, H, W)")

    assert lr_volume.shape[1] == 1, "Image must be greyscale"
    
    interpolation_dim_index = {"D" : 0, "H" : 1, "W" : 2}[interpolation_dim]


    lr_volume = lr_volume.to(device)
    model = model.to(device)

    B = lr_volume.shape[0]
    num_slices = lr_volume.shape[2+interpolation_dim_index]

    def _build_slice_selection_tuple(slcie_index ,interpolation_dim_index=0): 
        slice_selection_tuple = tuple(slcie_index if i == interpolation_dim_index else slice(None)  for i in range(3))
        return (slice(None),) + slice_selection_tuple

    hr_volumes_5d = []
    for batch_index in range(B): 
        lr_volume_4d = lr_volume[batch_index, :, :, :, :]
        
        hr_slices_4d = []
        for slice_index in range(num_slices):
            # Builds shape: (C, H, W)
            sclice_selection_3d = _build_slice_selection_tuple(slice_index, interpolation_dim_index)
            lr_slice_3d = lr_volume_4d[sclice_selection_3d]
            
            # Add Batch dimension
            hr_slice_3d = model(lr_slice_3d.unsqueeze(0))
            # Store and add dimension to enable concatenation later
            # +1 because one Channel dimension is before the spaial dimensions
            hr_slices_4d.append(hr_slice_3d.unsqueeze(1+interpolation_dim_index))
        
        hr_volume_4d = torch.cat(hr_slices_4d, dim=1+interpolation_dim_index)
        # Remove Channel dimension, bring to cpu and convert to numpy
        hr_volume_3d_np = hr_volume_4d.squeeze(0).detach().cpu().numpy()
        zoom_factors = [float(upscale_factor) if i == interpolation_dim_index else 1.0 for i in range(3)]
        hr_volume_3d_np = nd.zoom(hr_volume_3d_np, zoom=zoom_factors, oder=interpolation_order)
        
        hr_volume_4d = torch.from_numpy(hr_volume_3d_np).to(lr_volume.dtype).to(device).unsqueeze(0)
        hr_volumes_5d.append(hr_volume_4d.unsqueeze(0))
    hr_volumes_tensor_5d = torch.cat(hr_volumes_5d, dim=0)
    return hr_volumes_tensor_5d

if __name__ == "__main__": 
    import sys
    sys.path.append("/content/superresolution_3d/models/dense_net")
    from basic_efficient_dense_net_2d import BasicEfficientDenseNet2d
    model = BasicEfficientDenseNet2d()
    x = torch.randn(2, 1, 64, 32, 32)
    y = build_sclie_wise_hr_volume(
        lr_volume=x, 
        model=model, 
        interpolation_order=1, 
        interpolation_dim="D",
    )
    assert y.shape == (2, 1, 128, 64, 64), f"Wrong shape: {y.shape}"





    