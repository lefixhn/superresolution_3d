'''
    Uses a 2D SR Model to generate a 3D HR Volume
    Two axes are upscaled though the 2D Model. 
    The remaining axe is upscaled though interpolation
'''
import torch 

def build_sclie_wise_hr_volume(
    lr_volume: torch.Tensor, 
    model: callable, 
    interpolation_order=1, 
    interpolation_axe_index=0     # Can be 0, 1 or 2
    upscale_factor=2, 
    device = "cuda" if torch.cuda.is_available() else "cpu", 
): 
    if lr_volume.ndims() == 4: 
        lr_volume = lr_volume.unsqueeze(0)
    elif lr_volume.ndims() == 3: 
        lr_volume = lr_volume.unsqueeze(0).unsqueeze(0)
    elif not lr_volume.ndims() == 5: 
        raise TypeError("Input must have shape (B, C, D, H, W), (C, D, H, W) or (D, H, W)")

    assert lr_volume.shape[1] == 1, "Image must be greyscale"

    lr_volume = lr_volume.to(device)
    model = model.to(device)

    B = lr_volume.shape[0]
    num_slices = lr_volume.shape[2+interpolation_axe_index]

    def _build_slice_selection_tuple(sclie_index ,interpolation_axe_index=0): 
        slice_selection_tuple = tuple(sclie_index if i == interpolation_axe_index else slice(None)  for i in range(3))
        return (1,) + slice_selection_tuple

    for batch_index in range(B): 
        lr_volume_4d = lr_volume[batch_index, :, :, :, :]
        
        hr_slices_4d = []
        for slice_index in range(num_slices):
            # Shape: (C, H, W)
            lr_slice_3d = _build_slice_selection_tuple(slice_index, interpolation_axe_index)
            
            hr_slice_3d = model(lr_slice_3d)
            # Store and add dimension to enable concatenation later
            hr_slices_4d.append(hr_slice_3d.unsqueeze(1+interpolation_axe_index))
        
        hr_volume_4d = torch.cat(hr_slices_4d, dim=1+interpolation_axe_index)
        hr_volume_3d_np = torch.to_nump


    