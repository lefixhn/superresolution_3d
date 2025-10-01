'''
    Compares a 2D Model with a 3D Model 
    The 2D Model builds a 3D volumina though slice wise superresolution
    along two axes and interpolation along one axe


''' 
import sys
sys.path.append('/content/superresolution_3d/data_preprocessing')
sys.path.append('/content/superresolution_3d/evaluation')

import torch
torch.backends.cudnn.enabled = True
import numpy as np 
from tqdm import tqdm
import comparing_metrics as cm 
from typing import Dict, List, Callable
import build_slice_wise_hr_volume as swhrv
import image_degradation  as ideg
import importlib
importlib.reload(swhrv)

@torch.no_grad()
def compare_slice_wise_3d_with_3d(
    model: Callable,
    lr_hr_5d_tensor_tuples,
    autoprint=True, 
    device="cuda" if torch.cuda.is_available() else "cpu", 
) -> Dict[str, float]:

    if not isinstance(model, torch.nn.Module): 
        device = "cpu"

    mse_key = "mse"
    mae_key = "mae"
    psnr_key = "psnr"
    lpips_d_key = "lpips d"
    lpips_h_key = "lpips h"
    lpips_w_key = "lpips w"
    nmi_key = "nmi"

    results = {
        mse_key : 0.0,
        mae_key : 0.0,
        psnr_key : 0.0,
        lpips_d_key : 0.0,
        lpips_h_key : 0.0,
        lpips_w_key : 0.0,
        nmi_key : 0.0
    }
    
    if isinstance(model, torch.nn.Module):  
        model_3d = model_3d.to(device).float().eval()
        model_2d = model_2d.to(device).float().eval()

    for lr_volume_tensor_5d, hr_volume_tensor_5d in tqdm(lr_hr_5d_tensor_tuples, "Iterating trhough lr-hr-tuples"):
        # Bring to device
        lr_volume_tensor_5d = lr_volume_tensor_5d.to(device).float().contiguous()
        hr_volume_tensor_5d = hr_volume_tensor_5d.to(device).float().contiguous()
        # Generate SR
        sr_image_3d_model = model(lr_volume_tensor_5d)

        #Safety mechanisms
        # TODO: THIS CHECK SEEMS DANGEROUS AND NOT RIGHT TO ME
        assert sr_image_3d_model.shape == hr_volume_tensor_5d.shape[1:5]
        sr_image_3d_model = sr_image_3d_model.clamp(0,1).float()
        if sr_image_3d_model.ndim == 4:
            sr_image_3d_model = sr_image_3d_model.unsqueeze(0)


        # Apply metrics 
        results[mse_key] += cm.compare_mse(sr_image_3d_model ,hr_volume_tensor_5d)
        results[mae_key] += cm.compare_mae(sr_image_3d_model ,hr_volume_tensor_5d)
        results[psnr_key] += cm.compare_psnr(sr_image_3d_model ,hr_volume_tensor_5d)
        results[lpips_d_key] += cm.compare_lpips(sr_image_3d_model ,hr_volume_tensor_5d, compare_axis="D")
        results[lpips_h_key] += cm.compare_lpips(sr_image_3d_model ,hr_volume_tensor_5d, compare_axis="H")
        results[lpips_w_key] += cm.compare_lpips(sr_image_3d_model ,hr_volume_tensor_5d, compare_axis="W")
        results[nmi_key] += cm.compare_normalized_mutual_information(sr_image_3d_model ,hr_volume_tensor_5d)

    num_tuples = len(lr_hr_5d_tensor_tuples)

    results[mse_key] /= num_tuples
    results[mae_key] /= num_tuples
    results[psnr_key] /= num_tuples
    results[lpips_d_key] /= num_tuples
    results[lpips_h_key] /= num_tuples
    results[lpips_w_key] /= num_tuples
    results[nmi_key] /= num_tuples

    return results










'''

@torch.no_grad()
def compare_slice_wise_3d_with_3d(
    model_2d: torch.nn.Module, 
    model_3d: torch.nn.Module, 
    lr_hr_5d_tensor_tuples,
    interpolation_order=3, 
    autoprint=True, 
    device="cuda" if torch.cuda.is_available() else "cpu", 
) -> Dict[str, Dict[str, float]]:
    
    results = {}
    
    mean_mse_2d = 0.0
    mean_mae_2d = 0.0
    mean_psnr_2d = 0.0
    mean_nmi_2d = 0.0
    mean_lpips_d_2d = 0.0
    mean_lpips_h_2d = 0.0
    mean_lpips_w_2d = 0.0

    mean_mse_3d = 0.0
    mean_mae_3d = 0.0
    mean_psnr_3d = 0.0
    mean_nmi_3d = 0.0
    mean_lpips_d_3d = 0.0
    mean_lpips_h_3d = 0.0
    mean_lpips_w_3d = 0.0

    


    

    model_3d = model_3d.to(device).float().eval()
    model_2d = model_2d.to(device).float().eval()

    for lr_volume_tensor_5d, hr_volume_tensor_5d in tqdm(lr_hr_5d_tensor_tuples, "Iterating trhough lr-hr-tuples"):
        lr_volume_tensor_5d = lr_volume_tensor_5d.to(device).float().contiguous()
        hr_volume_tensor_5d = hr_volume_tensor_5d.to(device).float().contiguous()

        # Build SR Images
        sr_image_2d_model = swhrv.build_slice_wise_hr_volume(
            lr_volume=lr_volume_tensor_5d, 
            model=model_2d, 
            interpolation_order=interpolation_order, 
            interpolation_dim="D",
            device=device, 
        )
        sr_image_2d_model = sr_image_2d_model.to(device).float().contiguous()
        sr_image_3d_model = model_3d(lr_volume_tensor_5d)

        #Safety mechanisms
        sr_image_2d_model = sr_image_2d_model.clamp(0,1).float()
        sr_image_3d_model = sr_image_3d_model.clamp(0,1).float()
        if sr_image_3d_model.ndim == 4:
            sr_image_3d_model = sr_image_3d_model.unsqueeze(0)


        # Apply metrics 
        mean_mse_2d += cm.compare_mse(sr_image_2d_model ,hr_volume_tensor_5d)
        mean_mae_2d += cm.compare_mae(sr_image_2d_model ,hr_volume_tensor_5d)
        mean_psnr_2d += cm.compare_psnr(sr_image_2d_model ,hr_volume_tensor_5d)
        mean_nmi_2d += cm.compare_normalized_mutual_information(sr_image_2d_model ,hr_volume_tensor_5d)
        mean_lpips_d_2d += cm.compare_lpips(sr_image_2d_model ,hr_volume_tensor_5d, compare_axis="D")
        mean_lpips_h_2d += cm.compare_lpips(sr_image_2d_model ,hr_volume_tensor_5d, compare_axis="H")
        mean_lpips_w_2d += cm.compare_lpips(sr_image_2d_model ,hr_volume_tensor_5d, compare_axis="W")

        mean_mse_3d += cm.compare_mse(sr_image_3d_model ,hr_volume_tensor_5d)
        mean_mae_3d += cm.compare_mae(sr_image_3d_model ,hr_volume_tensor_5d)
        mean_psnr_3d += cm.compare_psnr(sr_image_3d_model ,hr_volume_tensor_5d)
        mean_nmi_3d += cm.compare_normalized_mutual_information(sr_image_3d_model ,hr_volume_tensor_5d)
        mean_lpips_d_3d += cm.compare_lpips(sr_image_3d_model ,hr_volume_tensor_5d, compare_axis="D")
        mean_lpips_h_3d += cm.compare_lpips(sr_image_3d_model ,hr_volume_tensor_5d, compare_axis="H")
        mean_lpips_w_3d += cm.compare_lpips(sr_image_3d_model ,hr_volume_tensor_5d, compare_axis="W")

    num_tuples = len(lr_hr_5d_tensor_tuples)
    
    

    name_model_2d = type(model_2d).__name__
    name_model_3d = type(model_3d).__name__
    results[name_model_2d]={}
    results[name_model_3d]={}


    results[name_model_2d]["mse"] = mean_mse_2d / num_tuples
    results[name_model_2d]["mae"] = mean_mae_2d / num_tuples
    results[name_model_2d]["psnr"] = mean_psnr_2d / num_tuples
    results[name_model_2d]["nmi"] = mean_nmi_2d / num_tuples
    results[name_model_2d]["lpips d"] = mean_lpips_d_2d / num_tuples
    results[name_model_2d]["lpips h"] = mean_lpips_h_2d / num_tuples
    results[name_model_2d]["lpips w"] = mean_lpips_w_2d / num_tuples

    results[name_model_3d]["mse"] = mean_mse_3d / num_tuples
    results[name_model_3d]["mae"] = mean_mae_3d / num_tuples
    results[name_model_3d]["psnr"] = mean_psnr_3d / num_tuples
    results[name_model_3d]["nmi"] = mean_nmi_3d / num_tuples
    results[name_model_3d]["lpips d"] = mean_lpips_d_3d / num_tuples
    results[name_model_3d]["lpips h"] = mean_lpips_h_3d / num_tuples
    results[name_model_3d]["lpips w"] = mean_lpips_w_3d / num_tuples
    

    if autoprint: 
        _print_compare_results(results)

    return results



def compare_sclice_wise_3d_with_3d_dense(
    checkpoint_path_2d, 
    checkpoint_path_3d, 
    num_images=150, 
    interpolation_order=3, 
): 
    results = {}
    sys.path.append("/content/superresolution_3d/models/dense_net")
    from basic_efficient_dense_net import BasicEfficientDenseNet
    from basic_efficient_dense_net_2d import BasicEfficientDenseNet2d

    model_2d = BasicEfficientDenseNet2d(num_dense_blocks=8, num_units_per_dense_block=8)
    model_3d = BasicEfficientDenseNet(num_dense_blocks=8, num_units_per_dense_block=8)
    sd2 = torch.load(checkpoint_path_2d, map_location="cpu")
    sd3 = torch.load(checkpoint_path_3d, map_location="cpu")
    sd2 = sd2.get("state_dict", sd2.get("model_state_dict", sd2))
    sd3 = sd3.get("state_dict", sd3.get("model_state_dict", sd3))

    model_2d.load_state_dict(sd2)
    model_3d.load_state_dict(sd3)

    degradation_models = build_degradation_models()
    # Load np volumes and convert to 5d tensor 
    hr_volumes_np = load_np_volumes(item_count=num_images)
    hr_volume_tensors_5d = [torch.from_numpy(hr_volume_np).float().unsqueeze(0).unsqueeze(0) for hr_volume_np in hr_volumes_np]

    for index, (degradation_model_name, degradation_model) in enumerate(degradation_models.items()): 
        lr_volume_tensors_5d = [degradation_model(hr_volume_tensor_5d) for hr_volume_tensor_5d in hr_volume_tensors_5d]
        lr_hr_volume_tensor_tuples = [(lr_volume_tensors_5d[i].float().contiguous(), hr_volume_tensors_5d[i].float().contiguous()) for i in range(len(lr_volume_tensors_5d))]

        print(f"##### DEG {degradation_model_name} #####")
        result=compare_slice_wise_3d_with_3d(
            model_2d=model_2d, 
            model_3d=model_3d, 
            lr_hr_5d_tensor_tuples=lr_hr_volume_tensor_tuples, 
            interpolation_order=interpolation_order, 
        )

        results[degradation_model_name] = result
    
    return results


def compare_on_training_degradation(
    checkpoint_path_2d, 
    checkpoint_path_3d, 
    num_images=100, 
    interpolation_order=3, 
): 
    results = {}
    sys.path.append("/content/superresolution_3d/datasets")
    import brats_dataset as ds
    sys.path.append("/content/superresolution_3d/models/dense_net")
    from basic_efficient_dense_net import BasicEfficientDenseNet
    from basic_efficient_dense_net_2d import BasicEfficientDenseNet2d

    dataset = ds.LazyLoadingBratsDataset(
        start_item_index=1100, 
        item_count=num_images, 
    )

    model_2d = BasicEfficientDenseNet2d(num_dense_blocks=8, num_units_per_dense_block=8)
    model_3d = BasicEfficientDenseNet(num_dense_blocks=8, num_units_per_dense_block=8)
    sd2 = torch.load(checkpoint_path_2d, map_location="cpu")
    sd3 = torch.load(checkpoint_path_3d, map_location="cpu")
    sd2 = sd2.get("state_dict", sd2.get("model_state_dict", sd2))
    sd3 = sd3.get("state_dict", sd3.get("model_state_dict", sd3))

    model_2d.load_state_dict(sd2)
    model_3d.load_state_dict(sd3)

    
    # Load np volumes and convert to 5d tensor 
    lr_hr_volume_tensor_tuples = [(lr_tensor_4d.unsqueeze(0), hr_tensor_4d.unsqueeze(0)) for lr_tensor_4d, hr_tensor_4d in dataset]
    

    results = compare_slice_wise_3d_with_3d(
        model_2d=model_2d, 
        model_3d=model_3d, 
        lr_hr_5d_tensor_tuples=lr_hr_volume_tensor_tuples, 
        interpolation_order=interpolation_order, 
    )
    
    
    return results

'''




##### Helper methods #####

def _print_compare_results(results: Dict[str, Dict[str, float]]): 
    for model_name, metric_results in results.items(): 
        model_text = str(model_name)
        for metric_name, average_metric_value in metric_results.items(): 
            model_text += f" | Average {metric_name}: {average_metric_value}"
        print(model_text)

def build_degradation_models(noise_sigmas: List[float]=[1.0/255.0, 12.0/255.0, 25.0/255.0], blur_sigmas: List[float]=[0.1, 1.2, 2.4], downscale_factor=2) -> Dict[str, Callable]: 
    ''' blur_sigmas and noise_sigmas must have the same length 
        builds as many degradations as the length of the list
        blur_sigmas and noise_sigmas at the same index will be combined to a 
        degradation model
    '''
    assert len(blur_sigmas) == len(noise_sigmas), "blur_sigmas and noise_sigmas must have the same length"
    degradation_models = {}

    for index in range(len(blur_sigmas)):
        blur_sigma, noise_sigma = blur_sigmas[index],  noise_sigmas[index]
        degradation_name = f"DEG[bs:{blur_sigma} | ns:{noise_sigma}]"
        degradation_models[degradation_name]= (
            lambda image, _noise_sigma=noise_sigma, _blur_sigma=blur_sigma: ideg.general_image_degradation_model(
                image=image, downscale_function=None ,downscale_factor=2, noise_sigma=_noise_sigma, blur_sigma=_blur_sigma
                )
        )

    return degradation_models

def load_np_volumes(
    start_index:int=1101, 
    item_count:int=10, 
    base_url: str = "/content/drive/MyDrive/superresolution_3d_data/datasets/advanced_degradation_t1/hr"
    ) -> List[np.ndarray]: 
    import os
    volumes = []
    entries = sorted([e for e in os.scandir(base_url) if e.name.endswith(".npy")], key=lambda e: e.name)

    for entry_index in range(start_index, min(len(entries), start_index+item_count)):
        entry = entries[entry_index]
        volumes.append(np.load(entry.path))
    return volumes