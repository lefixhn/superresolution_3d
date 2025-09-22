''' 
    This script provides methods to compare the performance of 
    a 2d and 3d model. It uses data with different degradation models
    and applies different metrics
'''
from typing import List, Tuple, Dict
import numpy as np
import sys
import torch 
import os
import importlib
sys.path.append('/content/superresolution_3d/data_preprocessing')
import image_degradation  as ideg
importlib.reload(ideg)
from tqdm import tqdm


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

def _build_degradated_np_volume_tuples(
    hr_volumes:List[np.ndarray], 
    noise_sigmas = [0, 12.0/255, 25.0/255.0], 
    blur_sigmas = [0.0, 1.2, 2.4]
) -> Dict[str, List[Tuple[np.ndarray, np.ndarray]]]: 
    ''' Returns multile lists of lr-hr-tuples. One list for each degradation. '''

    results = {}

    for noise_sigma in noise_sigmas: 
        for blur_sigma in blur_sigmas: 
            lr_hr_tuples = []
            for hr_volume in tqdm(
                hr_volumes,
                desc=f"Degrading bs={blur_sigma:.2f} ns={noise_sigma:.5f}",
                leave=False
            ):             
                # Degradate image with settings 
                lr_volume = ideg.general_image_degradation_model_on_3d_nparray(
                    image=hr_volume, 
                    noise_sigma=noise_sigma, 
                    blur_sigma=blur_sigma, 
                    downscale_function=ideg.cubic_downscale
                )
                lr_hr_tuples.append((lr_volume, hr_volume))
            # Add tuples to results dict
            degradation_name = f"Deg(bs={blur_sigma:.4f} | ns={noise_sigma:.4f})"
            results[degradation_name] = lr_hr_tuples
    return results

def _convert_np_slice_to_tensor(np_slice: np.ndarray) -> torch.Tensor:
    tensor = torch.as_tensor(np_slice, dtype=torch.float32).unsqueeze(0).unsqueeze(0) # Add Batch and channel dim 
    return tensor

def _convert_np_volume_tuple_to_tensor_list_for_2d(
    lr_volume: np.ndarray, 
    hr_volume: np.ndarray,
    step_length:int = 1 

) -> List[Tuple[torch.Tensor, torch.Tensor]]: 
    ''' Converts a 3d np volume tuple to multiple 2d slices in a list. 
    Each slice represented with a (1, 1, H, W) torch tensor.'''
    assert len(lr_volume.shape) == 3 and len(hr_volume.shape) == 3, "np volumes must be 3d"
    assert tuple(2*s for s in lr_volume.shape) == hr_volume.shape, f"lr shape must be half hr shape. lr: {lr_volume.shape} hr:{hr_volume.shape}"
    # Iterate though the orientations in 3d
    def _get_slice_selection(orienteation_index, iteration_axe_index) -> Tuple: 
        return tuple(iteration_axe_index if i == orienteation_index else slice(None) for i in range(3))
    lr_hr_tuples = []
    for orienteation_index in range(len(lr_volume.shape)): 
        iteration_axe_lr_size = lr_volume.shape[orienteation_index]
        for iteration_axe_lr_index in range(0, iteration_axe_lr_size, step_length):
            iteration_axe_hr_index = 2*iteration_axe_lr_index
            # Get selections 
            lr_selection = _get_slice_selection(orienteation_index, iteration_axe_lr_index)
            hr_selection_lower = _get_slice_selection(orienteation_index, iteration_axe_hr_index)
            hr_selection_upper = _get_slice_selection(orienteation_index, iteration_axe_hr_index+1)
            # Get slices 
            lr_slice = lr_volume[lr_selection]
            hr_slice = (hr_volume[hr_selection_lower] + hr_volume[hr_selection_upper]) / 2 # Average of both corresponding hr slices 
            # Convert to tensors 
            lr_tensor, hr_tensor = _convert_np_slice_to_tensor(lr_slice), _convert_np_slice_to_tensor(hr_slice)
            # Store tuple 
            lr_hr_tuples.append((lr_tensor, hr_tensor))
    return lr_hr_tuples


def _convert_np_volume_tuple_to_tensor(lr_volume, hr_volume: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    ''' Converts a 3d np volume tuple to a tensor tuple''' 
    return torch.as_tensor(lr_volume, dtype=torch.float32).unsqueeze(0).unsqueeze(0), torch.as_tensor(hr_volume).unsqueeze(0).unsqueeze(0)

def compare_2d_3d_models(
    models_2d : Dict[str, callable],   # Name -> 2D-Modell (expects (B,1,H,W) -> (B,1,H,W))
    models_3d : Dict[str, callable],   # Name -> 3D-Modell (expects (B,1,D,H,W) -> (B,1,D,H,W))
    hr_np_volumes : List[np.ndarray], 
    with_ssim=True,                    # optional, wird versucht (skimage), sonst übersprungen
    with_lpips=True                    # optional, wird versucht (lpips), sonst übersprungen
) -> Dict[str, Dict[str, Dict[str, float]]]:  # Degradation -> Model -> Metric -> float
    '''
        Compares 2d and 3d models on different degradations with MSE, MAE, PSNR (+ optional SSIM, LPIPS).
        Rückgabeformat:
        results[degradation_key][model_name]['mse'|'mae'|'psnr'|'ssim'|'lpips'] = float
    '''
    # Hilfs-Metriken (rein innerhalb der Funktion, keine neuen globalen Methoden)
    import math

    def _to_cpu_numpy(t: torch.Tensor) -> np.ndarray:
        return t.detach().float().cpu().numpy()

    def _mse(a: torch.Tensor, b: torch.Tensor) -> float:
        return torch.mean((a - b) ** 2).item()

    def _mae(a: torch.Tensor, b: torch.Tensor) -> float:
        return torch.mean(torch.abs(a - b)).item()

    def _psnr(a: torch.Tensor, b: torch.Tensor) -> float:
        mse = _mse(a, b)
        if mse <= 1e-12:
            return 99.0
        # Dynamischer Datenbereich aus HR ableiten (robust, falls nicht [0,1])
        hr = b
        data_range = (hr.max() - hr.min()).item()  # Hinweis: item() synchronisiert auf CUDA
        if data_range <= 0.0:
            data_range = 1.0
        return 20.0 * math.log10(data_range) - 10.0 * math.log10(mse)

    # Optional SSIM (skimage), optional LPIPS (lpips)
    have_ssim = False
    have_lpips = False
    ssim_module = None
    lpips_model = None

    if with_ssim:
        try:
            from skimage.metrics import structural_similarity as ssim_fn
            ssim_module = ssim_fn
            have_ssim = True
        except Exception:
            have_ssim = False  # skimage nicht vorhanden → wir lassen SSIM weg

    if with_lpips:
        try:
            import lpips  # pip package "lpips"
            lpips_model = lpips.LPIPS(net='alex')
            lpips_model.eval()
            have_lpips = True
        except Exception:
            have_lpips = False  # lpips nicht vorhanden → lassen wir weg

    # Degradationssätze bauen: {degradation_key: [(lr_np, hr_np), ...], ...}
    degradation_sets = _build_degradated_np_volume_tuples(hr_np_volumes)

    results: Dict[str, Dict[str, Dict[str, float]]] = {}

    # Durch alle Degradations-Einstellungen iterieren
    for degradation_key, lr_hr_list in degradation_sets.items():
        # Pro Degradation sammeln wir Metriken je Modellname
        results[degradation_key] = {}

        # ===== 3D-Modelle =====
        for model_name, model in models_3d.items():
            metric_sums = {'mse': 0.0, 'mae': 0.0, 'psnr': 0.0}
            if have_ssim:  metric_sums['ssim']  = 0.0
            if have_lpips: metric_sums['lpips'] = 0.0
            n_items = 0

            # Device des Modells bestimmen (CPU oder CUDA)
            mdev = next(model.parameters()).device
            model.eval()

            with torch.no_grad():
                for (lr_np, hr_np) in tqdm(
                    lr_hr_list,
                    desc=f"[3D] {model_name} | {degradation_key}",
                    leave=False
                ):
                    # (B,C,D,H,W)
                    lr_t, hr_t = _convert_np_volume_tuple_to_tensor(lr_np, hr_np)  # deine Methode
                    lr_t = lr_t.to(mdev, non_blocking=True)
                    hr_t = hr_t.to(mdev, non_blocking=True)

                    # schnelleres Inferenz-Compute auf CUDA
                    use_amp = (mdev.type == 'cuda')
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        sr_t = model(lr_t)

                    # Metriken über komplettes Volumen
                    metric_sums['mse']  += _mse(sr_t, hr_t)
                    metric_sums['mae']  += _mae(sr_t, hr_t)
                    metric_sums['psnr'] += _psnr(sr_t, hr_t)

                    if have_ssim:
                        # SSIM 3D nicht trivial → durchschnitt der Slice-SSIMs über die Tiefenachse
                        # Wir nehmen hier die "axiale" Richtung (D) und mitteln SSIM in 2D-Ebene.
                        sr_np = _to_cpu_numpy(sr_t)[0,0]  # (D,H,W)
                        hr_np_v = _to_cpu_numpy(hr_t)[0,0]
                        D = sr_np.shape[0]
                        ssim_acc = 0.0
                        for d in range(D):
                            try:
                                data_range = (hr_np_v[d].max() - hr_np_v[d].min()) or 1.0
                                ssim_val = ssim_module(hr_np_v[d], sr_np[d], data_range=data_range)
                            except Exception:
                                ssim_val = 0.0
                            ssim_acc += float(ssim_val)
                        metric_sums['ssim'] += (ssim_acc / max(D,1))

                    if have_lpips:
                        # LPIPS erwartet (N,3,H,W). Wir duplizieren den 1-Kanal auf 3 Kanäle und mitteln über D.
                        sr_np = _to_cpu_numpy(sr_t)[0,0]  # (D,H,W)
                        hr_np_v = _to_cpu_numpy(hr_t)[0,0]
                        D = sr_np.shape[0]
                        lpips_acc = 0.0
                        for d in range(D):
                            sr_img = torch.from_numpy(sr_np[d]).float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
                            hr_img = torch.from_numpy(hr_np_v[d]).float().unsqueeze(0).unsqueeze(0)
                            # auf 3 Kanäle kopieren:
                            sr_img3 = sr_img.repeat(1,3,1,1)
                            hr_img3 = hr_img.repeat(1,3,1,1)
                            try:
                                lp = lpips_model(sr_img3, hr_img3).item()
                            except Exception:
                                lp = 0.0
                            lpips_acc += lp
                        metric_sums['lpips'] += (lpips_acc / max(D,1))

                    n_items += 1

            # Mittelwerte
            denom = max(n_items, 1)
            results[degradation_key][model_name] = {k: (v/denom) for k,v in metric_sums.items()}

        # ===== 2D-Modelle =====
        for model_name, model in models_2d.items():
            metric_sums = {'mse': 0.0, 'mae': 0.0, 'psnr': 0.0}
            if have_ssim:  metric_sums['ssim']  = 0.0
            if have_lpips: metric_sums['lpips'] = 0.0
            n_slices_total = 0

            # Device des Modells bestimmen
            mdev = next(model.parameters()).device
            model.eval()

            with torch.no_grad():
                for (lr_np, hr_np) in tqdm(
                    lr_hr_list,
                    desc=f"[2D] {model_name} | {degradation_key}",
                    leave=False
                ):
                    # In 2D-Slices zerlegen (alle Orientierungen, deine Mittelungslogik für HR)
                    slice_pairs = _convert_np_volume_tuple_to_tensor_list_for_2d(lr_np, hr_np, step_length=1)  # deine Methode
                    for (lr_slice_t, hr_slice_t) in slice_pairs:
                        lr_slice_t = lr_slice_t.to(mdev, non_blocking=True)
                        hr_slice_t = hr_slice_t.to(mdev, non_blocking=True)

                        use_amp = (mdev.type == 'cuda')
                        with torch.cuda.amp.autocast(enabled=use_amp):
                            sr_slice_t = model(lr_slice_t)

                        # Metriken slice-weise
                        metric_sums['mse']  += _mse(sr_slice_t, hr_slice_t)
                        metric_sums['mae']  += _mae(sr_slice_t, hr_slice_t)
                        metric_sums['psnr'] += _psnr(sr_slice_t, hr_slice_t)

                        if have_ssim:
                            try:
                                hr_np2 = _to_cpu_numpy(hr_slice_t)[0,0]  # (H,W)
                                sr_np2 = _to_cpu_numpy(sr_slice_t)[0,0]
                                data_range = (hr_np2.max() - hr_np2.min()) or 1.0
                                ssim_val = ssim_module(hr_np2, sr_np2, data_range=data_range)
                            except Exception:
                                ssim_val = 0.0
                            metric_sums['ssim'] += float(ssim_val)

                        if have_lpips:
                            # (1,1,H,W) -> (1,3,H,W)
                            sr_img3 = sr_slice_t.repeat(1,3,1,1)
                            hr_img3 = hr_slice_t.repeat(1,3,1,1)
                            try:
                                lp = lpips_model(sr_img3, hr_img3).item()
                            except Exception:
                                lp = 0.0
                            metric_sums['lpips'] += lp

                        n_slices_total += 1

            denom = max(n_slices_total, 1)
            results[degradation_key][model_name] = {k: (v/denom) for k,v in metric_sums.items()}

    return results


def print_results_nested(results: Dict[str, Dict[str, Dict[str, float]]]) -> None:
    """
    Iteriert automatisch über alle Degradations-Keys, Modelle und Metriken und druckt die Ergebnisse.
    Ausgabeform:
    Degradation-Key
      - Model-Name:
          mse=..., mae=..., psnr=..., [ssim=...], [lpips=...]
    """
    for degradation_key, models_dict in results.items():
        print(f"\n=== {degradation_key} ===")
        for model_name, metrics_dict in models_dict.items():
            # stabil sortieren: erst "klassische" Metriken
            ordered_keys = ['mse','mae','psnr'] + [k for k in metrics_dict.keys() if k not in ('mse','mae','psnr')]
            metrics_str = ", ".join([f"{k}={metrics_dict[k]:.6f}" for k in ordered_keys if k in metrics_dict])
            print(f"  - {model_name}: {metrics_str}")

