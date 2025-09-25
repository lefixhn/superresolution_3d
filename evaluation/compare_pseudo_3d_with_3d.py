'''
    Compares a 2D Model with a 3D Model 
    The 2D Model builds a 3D volumina though slice wise superresolution
    along two axes and interpolation along one axe


''' 
import torch
from tqdm import tqdm
import comparing_metrics as cm 
from typing import Dict, List, Callable
from build_sclie_wise_hr_volume import build_sclie_wise_hr_volume
import sys
sys.path.append('/content/superresolution_3d/data_preprocessing')
import importlib
import image_degradation  as ideg

@torch.no_grad()
def compare_pseudo_3d_with_3d(
    model_2d: torch.nn.Module, 
    model_3d: torch.nn.Module, 
    lr_hr_5d_tensor_tuples,
    autoprint=True, 
    device="cuda" if torch.cuda.is_available() else "cpu", 
) -> Dict[str, Dict[str, float]]:
    
    results = {}
    
    mean_mse_2d = 0.0
    mean_mae_2d = 0.0
    mean_psnr_2d = 0.0

    mean_mse_3d = 0.0
    mean_mae_3d = 0.0
    mean_psnr_3d = 0.0
    

    model_3d = model_3d.to(device)

    for lr_volume_tensor_5d, hr_volume_tensor_5d in tqdm(lr_hr_5d_tensor_tuples, "Iterating trhough lr-hr-tuples"):
        # Build SR Images
        sr_image_2d_model = build_sclie_wise_hr_volume(
            lr_volume=lr_volume_tensor_5d, 
            model: model_2d, 
            interpolation_dim="D",
            device=device, 
        )

        sr_image_3d_model = model_3d(lr_volume_tensor_5d)

        # Apply metrics 
        mean_mse_2d += cm.compare_mse(sr_image_2d_model ,hr_volume_tensor_5d)
        mean_mse_2d += cm.compare_mae(sr_image_2d_model ,hr_volume_tensor_5d)
        mean_mse_2d += cm.compare_psnr(sr_image_2d_model ,hr_volume_tensor_5d)

        mean_mse_3d += cm.compare_mse(sr_image_3d_model ,hr_volume_tensor_5d)
        mean_mse_3d += cm.compare_mae(sr_image_3d_model ,hr_volume_tensor_5d)
        mean_mse_3d += cm.compare_psnr(sr_image_3d_model ,hr_volume_tensor_5d)

    num_tuples = len(lr_hr_5d_tensor_tuples)
    
    mean_mse_2d /= num_tuples
    mean_mae_2d /= num_tuples
    mean_psnr_2d /= num_tuples

    mean_mse_3d = /= num_tuples
    mean_mae_3d = /= num_tuples
    mean_psnr_3d = /= num_tuples

    name_model_2d = type(model_2d).__name__
    name_model_3d = type(model_3d).__name__


    results[name_model_2d]["mse"] = mean_mse_2d
    results[name_model_2d]["mae"] = mean_mse_2d
    results[name_model_2d]["psnr"] = mean_mse_2d

    results[name_model_3d]["mse"] = mean_mse_3d
    results[name_model_3d]["mae"] = mean_mse_3d
    results[name_model_3d]["psnr"] = mean_mse_3d

    if autoprint: 
        _print_compare_results(results)

    return results



def compare_pseudo_3d_dense_with_3d(
    checkpoint_path_2d, 
    checkpoint_path_3d, 
    num_images=150, 
): 
    results = {}

    from basic_efficient_dense_net import BasicEfficientDenseNet
    from basic_efficient_dense_net_2d import BasicEfficientDenseNet2d

    model_2d = BasicEfficientDenseNet2d(num_dense_blocks=8, num_units_per_dense_block=8)
    model_3d = BasicEfficientDenseNet(num_dense_blocks=8, num_units_per_dense_block=8)

    degradation_models = build_degradation_models()
    # Load np volumes and convert to 5d tensor 
    hr_volumes_np = load_np_volumes(item_count=num_images)
    hr_volume_tensors_5d = [torch.from_numpy(hr_volume_np).unsqueeze(0).unsqueeze(0) for hr_volume_np in hr_volumes_np]

    for index, degradation_model_name, degradation_model in degradation_models.items(): 
        lr_volume_tensors_5d = [degradation_model(hr_volume_tensor_5d) for hr_volume_tensor_5d in hr_volume_tensors_5d]
        lr_hr_volume_tensor_tuples = (lr_volume_tensors_5d[i], hr_volume_tensors_5d[i] for i in range(len(lr_volume_tensors_5d)))

        print(f"##### DEG {degradation_model_name} "#####")
        result=compare_pseudo_3d_with_3d(
            model_2d=model_2d, 
            model_3d=model_3d, 
            lr_hr_5d_tensor_tuples=lr_hr_volume_tensor_tuples, 
        )

        results[degradation_model_name] = result
    
    return results





### Helper methods

def _print_compare_results(results: Dict[str, Dict[str, float]]): 
    for model_name, metric_results in results.items(): 
        model_text = str(model_name)
        for metric_name, average_metric_value in metric_results.items(): 
            model_text += f" | Average {metric_name}: {average_metric_value}"

def build_degradation_models(blur_sigmas: List[float]=[12.0/255.0, 12.0/255.0, 25.0/255.0], noise_sigmas: List[float]=[0.1, 1.2, 2.4], downscale_factor=2) -> Dict[str, Callable]: 
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
    volumes = []
    entries = sorted([e for e in os.scandir(base_url) if e.name.endswith(".npy")], key=lambda e: e.name)

    for entry_index in range(start_index, min(len(entries), start_index+item_count)):
        entry = entries[entry_index]
        volumes.append(np.load(entry.path))
    return volumes