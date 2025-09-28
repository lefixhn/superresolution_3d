'''
    This script was AI generated
'''


# --- 3D NPY Volume Viewer (no axis reordering) ---
# Usage:
# 1) Set BASE_DIR to your folder with .npy volumes (each shaped [D,H,W])
# 2) Run the cell -> choose file, axis (D/H/W), then navigate with slider/prev/next

import os
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import Dropdown, ToggleButtons, IntSlider, Button, HBox, VBox, Checkbox, Label
from IPython.display import display, clear_output

BASE_DIR = "/content/drive/MyDrive/superresolution_3d_data/datasets/advanced_degradation_t1/hr"  # <--- anpassen

# Collect .npy files
files = [f for f in sorted(os.listdir(BASE_DIR)) if f.endswith(".npy")]
if not files:
    raise FileNotFoundError(f"Keine .npy Dateien in {BASE_DIR} gefunden.")

# Widgets
w_file   = Dropdown(options=files, description="File:", layout={'width':'50%'})
w_axis   = ToggleButtons(options=["D","H","W"], description="Axis:", value="D")
w_slice  = IntSlider(value=0, min=0, max=1, step=1, description="Slice:", continuous_update=False)
w_prev   = Button(description="◀︎ Prev", tooltip="previous slice")
w_next   = Button(description="Next ▶︎", tooltip="next slice")
w_autops = Checkbox(value=True, description="Auto-contrast per-slice")
w_autogl = Checkbox(value=False, description="Auto-contrast global")
w_info   = Label()

# State
_cache = {
    "arr": None,
    "vmin": None,
    "vmax": None,
    "shape": None,
}

def _load(path):
    # mmap=read-only; we’ll materialize later only if needed
    arr = np.load(path, mmap_mode="r")
    if arr.ndim != 3:
        raise ValueError(f"Erwarte 3D-Array [D,H,W], bekam shape={arr.shape} in {os.path.basename(path)}")
    return arr

def _axis_len(arr, axis):
    if axis == "D": return arr.shape[0]
    if axis == "H": return arr.shape[1]
    if axis == "W": return arr.shape[2]
    raise ValueError(axis)

def _get_slice(arr, axis, s):
    # IMPORTANT: no transpose, just direct indexing
    if axis == "D":
        return np.asarray(arr[s, :, :])
    elif axis == "H":
        return np.asarray(arr[:, s, :])
    elif axis == "W":
        return np.asarray(arr[:, :, s])

def _compute_global_vrange(arr, axis):
    # For consistent visualization across slices (optional)
    # Use percentiles to be robust to outliers
    # Flatten full volume for percentiles (cheap) – does not alter array
    data = np.asarray(arr)  # materialize once
    lo, hi = np.percentile(data, [1.0, 99.0])
    if lo >= hi:  # fallback
        lo, hi = float(np.min(data)), float(np.max(data))
    return float(lo), float(hi)

def _update_slice_range():
    arr = _cache["arr"]
    axis = w_axis.value
    n = _axis_len(arr, axis)
    w_slice.max = max(n - 1, 0)
    w_slice.value = min(w_slice.value, w_slice.max)

def _render():
    arr = _cache["arr"]
    axis = w_axis.value
    s = w_slice.value

    sl = _get_slice(arr, axis, s)

    # Determine contrast range
    if w_autogl.value and (_cache["vmin"] is not None):
        vmin, vmax = _cache["vmin"], _cache["vmax"]
    elif w_autogl.value and (_cache["vmin"] is None):
        vmin, vmax = _compute_global_vrange(arr, axis)
        _cache["vmin"], _cache["vmax"] = vmin, vmax
    elif w_autops.value:
        lo, hi = np.percentile(sl, [1.0, 99.0])
        if lo >= hi:
            lo, hi = float(np.min(sl)), float(np.max(sl))
        vmin, vmax = float(lo), float(hi)
    else:
        vmin, vmax = None, None  # raw

    plt.figure(figsize=(5,5))
    plt.imshow(sl, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
    plt.title(f"{w_file.value} | Axis={axis} | Slice={s} | shape={arr.shape}")
    plt.axis("off")
    plt.show()

    # Info label
    plane = {"D":"H×W", "H":"D×W", "W":"D×H"}[axis]
    w_info.value = f"Array shape [D,H,W]={arr.shape} | Bildebene={plane} | Anzeige vmin={vmin} vmax={vmax}"

def _on_change_file(*_):
    # Load new file, reset cache + slider
    path = os.path.join(BASE_DIR, w_file.value)
    _cache["arr"] = _load(path)
    _cache["vmin"], _cache["vmax"] = None, None
    _cache["shape"] = _cache["arr"].shape
    _update_slice_range()
    clear_output(wait=True)
    display(ui)
    _render()

def _on_change_axis(*_):
    # Reset slice slider range & maybe recompute global vrange
    _update_slice_range()
    if w_autogl.value:
        _cache["vmin"], _cache["vmax"] = None, None
    clear_output(wait=True)
    display(ui)
    _render()

def _on_change_slice(*_):
    clear_output(wait=True)
    display(ui)
    _render()

def _on_prev(_):
    if w_slice.value > 0:
        w_slice.value -= 1

def _on_next(_):
    if w_slice.value < w_slice.max:
        w_slice.value += 1

def _on_toggle_contrast(_):
    # If switching global/local, reset cached global v-range
    if w_autogl.value:
        _cache["vmin"], _cache["vmax"] = None, None
    clear_output(wait=True)
    display(ui)
    _render()

# Wire events
w_file.observe(_on_change_file, names="value")
w_axis.observe(_on_change_axis, names="value")
w_slice.observe(_on_change_slice, names="value")
w_prev.on_click(_on_prev)
w_next.on_click(_on_next)
w_autops.observe(_on_toggle_contrast, names="value")
w_autogl.observe(_on_toggle_contrast, names="value")

# Layout
ui = VBox([
    HBox([w_file]),
    HBox([w_axis, w_slice, w_prev, w_next]),
    HBox([w_autops, w_autogl]),
    w_info
])

display(ui)
# initial load
_on_change_file()