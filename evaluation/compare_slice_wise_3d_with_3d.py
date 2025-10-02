'''
    Compares a 2D Model with a 3D Model and tricubic interpolation
    The 2D Model builds a 3D volumina though slice wise superresolution
    along two axes and interpolation along one axe

    Use compare_models to compare

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
def evaluate_model(
    model: Callable,
    lr_hr_5d_tensor_tuples,
    autoprint=True, 
    device="cuda" if torch.cuda.is_available() else "cpu", 
) -> Dict[str, float]:
    assert len(lr_hr_5d_tensor_tuples) > 0, "tuples cannot be empty"

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
        model = model.to(device).float().eval()
        

    for lr_volume_tensor_5d, hr_volume_tensor_5d in tqdm(lr_hr_5d_tensor_tuples, desc="Iterating trhough lr-hr-tuples"):
        # Bring to device
        lr_volume_tensor_5d = lr_volume_tensor_5d.to(device).float().contiguous()
        hr_volume_tensor_5d = hr_volume_tensor_5d.to(device).float().contiguous()
        # Generate SR
        sr_image_3d_model = model(lr_volume_tensor_5d)

        #Safety mechanisms
        # TODO: THIS CHECK SEEMS DANGEROUS AND NOT RIGHT TO ME
        assert sr_image_3d_model.shape[-3:] == hr_volume_tensor_5d.shape[-3:], f"Shape mismatch in last 3 dimensions {sr_image_3d_model.shape[-3:]} should equal {hr_volume_tensor_5d.shape[-3:]}"

        sr_image_3d_model = sr_image_3d_model.clamp(0,1).float()
        if sr_image_3d_model.ndim == 4:
            sr_image_3d_model = sr_image_3d_model.unsqueeze(0)
        
        sr_image_3d_model = sr_image_3d_model.to(hr_volume_tensor_5d.device)
        sr_image_3d_model = sr_image_3d_model.to(hr_volume_tensor_5d.dtype)


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

    if autoprint: 
        for metric, value in results.items(): 
            print(f"METRIC {metric} : {value}")
        print("########################")

    return results


def evaluate_models_on_degradation(
    models: Dict[str, Callable], 
    lr_hr_5d_tensor_tuples,
) -> Dict[str, Dict[str, float]]:
    '''
        Call with 
        results[model_name][metric_name]
    '''
    results = {}

    for model_name, model in models.items(): 
        results[model_name] = evaluate_model(
            model=model,
            lr_hr_5d_tensor_tuples=lr_hr_5d_tensor_tuples,
            autoprint=False, 
        )
    
    return results



def compare_models(
    checkpoint_path_2d, 
    checkpoint_path_3d, 
    num_images=100, 
    sclice_wise_interpolation_order=3, 
    autoprint=True, 
) -> Dict[str, Dict[str, Dict[str, float]]]: 
    '''
        Call with results[degradation_name][model_name][metric_name]
    '''

    results = {}
    
    # Build 3D and 2D Model 
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
    # Bring  models to device and trainingmode
    run_device = "cuda" if torch.cuda.is_available() else "cpu"
    model_2d = model_2d.to(run_device).float().eval()
    model_3d = model_3d.to(run_device).float().eval()
    slice_wise_model = lambda tensor_5d: swhrv.build_slice_wise_hr_volume(
        lr_volume = tensor_5d.to(run_device), 
        model=model_2d, 
        interpolation_order=sclice_wise_interpolation_order, 
        interpolation_dim="D", 
        upscale_factor=2, 
        device=run_device,
    ) 

    sys.path.append("/content/superresolution_3d/evaluation")
    from interpolation_3d import interpolation_3d
    interpolation_model = lambda tensor_5d: interpolation_3d(
        tensor_5d=tensor_5d, 
        order=3, 
        upscale_factor=2
    )

    print("2D training mode? ", model_2d.training)  # sollte False sein
    print("3D training mode? ", model_3d.training)

    models = {
        "3D-MODEL" : model_3d, 
        "2D-MODEL SLICE-WISE" : slice_wise_model, 
        "TRICUBIC INTERPOLATION" : interpolation_model
    }


    degradation_models = build_degradation_models()
    # Load np volumes and convert to 5d tensor 
    hr_volumes_np = load_np_volumes(item_count=num_images)
    hr_volume_tensors_5d = [torch.from_numpy(hr_volume_np).float().unsqueeze(0).unsqueeze(0) for hr_volume_np in hr_volumes_np]

    # Iterate through static degradation models
    for index, (degradation_model_name, degradation_model) in enumerate(degradation_models.items()): 
        # Build tuples from degradation 
        lr_volume_tensors_5d = [degradation_model(hr_volume_tensor_5d) for hr_volume_tensor_5d in hr_volume_tensors_5d]
        lr_hr_volume_tensor_tuples = [(lr_volume_tensors_5d[i].float().contiguous(), hr_volume_tensors_5d[i].float().contiguous()) for i in range(len(lr_volume_tensors_5d))]
        # Store results of degradationmodel 
        results[degradation_model_name] = evaluate_models_on_degradation(
            models = models, 
            lr_hr_5d_tensor_tuples=lr_hr_volume_tensor_tuples
        )
    
    # Compare on random degradation model 
    # Load np volumes and convert to 5d tensor 
    sys.path.append("/content/superresolution_3d/datasets")
    import brats_dataset as ds
    dataset = ds.LazyLoadingBratsDataset(
        start_item_index=1100, 
        item_count=num_images, 
    )
    lr_hr_volume_tensor_tuples = [(lr_tensor_4d.unsqueeze(0), hr_tensor_4d.unsqueeze(0)) for lr_tensor_4d, hr_tensor_4d in dataset]
    results["TRAINING DEGRADATION MODEL"] = evaluate_models_on_degradation(
            models = models, 
            lr_hr_5d_tensor_tuples=lr_hr_volume_tensor_tuples
        )

    if autoprint: 
        for degradation_name, degradation_results in results.items(): 
            print(f"DEGRADATION MODEL: {degradation_name}")
            for model_name, model_results in degradation_results.items(): 
                print(f"MODEL: {model_name}")
                for metic_name, metric_value in model_results.items():
                    mt = " " 
                    print(f"METRIC: {metic_name} {mt * (10-len(metic_name))} : {metric_value}")

    return results
    



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