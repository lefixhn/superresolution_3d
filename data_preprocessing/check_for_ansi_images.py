#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import math
from typing import Tuple, List, Optional
from collections import Counter
# BraTS2021
# SEARCH_PATH = "/content/drive/MyDrive/superresolution_3d_data/datasets/BraTS2021_Training_Data"
# IXI 
#SEARCH_PATH = "/content/drive/MyDrive/superresolution_3d_data/datasets/IXI-T1"
# fastMRI
SEARCH_PATH = "/content/drive/MyDrive/superresolution_3d_data/datasets/multicoil_train"
print("analyzing data")

try:
    import nibabel as nib
except ImportError:
    print("Fehler: 'nibabel' ist nicht installiert. Installiere es mit: pip install nibabel")
    sys.exit(1)

try:
    import h5py
except ImportError:
    print("Fehler: 'h5py' ist nicht installiert. Installiere es mit: pip install h5py")
    sys.exit(1)

import xml.etree.ElementTree as ET


def is_isotropic(zooms: Tuple[float, ...], rel_tol: float = 1e-3, abs_tol: float = 1e-6) -> Optional[bool]:
    """
    True/False: Isotropie beurteilbar, None: nicht anwendbar (z.B. nur 2D-Spacings bekannt).
    """
    if len(zooms) < 3 or any(z is None for z in zooms[:3]):
        return None
    z0, z1, z2 = zooms[:3]
    return (math.isclose(z0, z1, rel_tol=rel_tol, abs_tol=abs_tol) and
            math.isclose(z1, z2, rel_tol=rel_tol, abs_tol=abs_tol))


def get_nifti_zooms_and_shape(path: str) -> Tuple[Tuple[float, ...], Tuple[int, ...]]:
    img = nib.load(path)
    zooms = tuple(img.header.get_zooms())
    shape = img.shape
    # auf 3D-konsistente Darstellung kürzen
    if len(shape) >= 3:
        shape3 = tuple(shape[:3])
    else:
        shape3 = tuple(shape)
    return zooms, shape3


def _safe_float(text: Optional[str]) -> Optional[float]:
    try:
        return float(text) if text is not None else None
    except Exception:
        return None


def get_fastmri_h5_spacings_and_shape(path: str) -> Tuple[Tuple[Optional[float], ...], Tuple[int, ...]]:
    """
    Liest fastMRI .h5:
      - nutzt bevorzugt 'kspace', fällt auf 'reconstruction_rss' (Magnitude-Recon) zurück
      - shape: bei kspace -> (H, W); bei reconstruction_rss -> (H, W)
      - spacings: aus ismrmrd_header (x,y); z bleibt bei 2D None
    """
    with h5py.File(path, "r") as f:
        dataset = None
        if "kspace" in f:
            dataset = f["kspace"]
            # (num_slices, num_coils, H, W) -> (H, W)
            if dataset.ndim >= 4:
                shape2 = (int(dataset.shape[-2]), int(dataset.shape[-1]))
            else:
                raise RuntimeError("Unerwartete kspace-Form.")
        elif "reconstruction_rss" in f:
            dataset = f["reconstruction_rss"]
            # (num_slices, H, W) -> (H, W)
            if dataset.ndim >= 3:
                shape2 = (int(dataset.shape[-2]), int(dataset.shape[-1]))
            else:
                raise RuntimeError("Unerwartete reconstruction_rss-Form.")
        else:
            raise RuntimeError("Kein 'kspace' und kein 'reconstruction_rss' im HDF5 gefunden.")

        # Spacings aus ismrmrd_header (falls vorhanden)
        sx = sy = sz = None
        hdr_xml = f.attrs.get("ismrmrd_header", None)
        if hdr_xml is not None:
            try:
                root = ET.fromstring(hdr_xml)
                enc = root.find(".//encoding")
                rs = enc.find("reconSpace") if enc is not None else None
                ms = rs.find("matrixSize") if rs is not None else None
                fov = rs.find("fieldOfView_mm") if rs is not None else None

                nx = _safe_float(ms.findtext("x")) if ms is not None else None
                ny = _safe_float(ms.findtext("y")) if ms is not None else None
                fx = _safe_float(fov.findtext("x")) if fov is not None else None
                fy = _safe_float(fov.findtext("y")) if fov is not None else None

                sx = (fx / nx) if (fx and nx and nx > 0) else None
                sy = (fy / ny) if (fy and ny and ny > 0) else None
            except Exception:
                pass

        zooms = (sx, sy, sz)  # z bleibt None (2D)
        return zooms, shape2

def get_zooms_and_shape(path: str) -> Tuple[Tuple[Optional[float], ...], Tuple[int, ...], str]:
    """
    Dispatcher je nach Dateiendung. Gibt (zooms, shape, typ) zurück.
    typ in {"nifti","fastmri-h5"}
    """
    low = path.lower()
    if low.endswith(".nii.gz") or low.endswith(".nii"):
        z, s = get_nifti_zooms_and_shape(path)
        return z, s, "nifti"
    elif low.endswith(".h5"):
        z, s = get_fastmri_h5_spacings_and_shape(path)
        return z, s, "fastmri-h5"
    else:
        raise ValueError(f"Unbekanntes Format: {path}")


def find_files(root: str) -> List[str]:
    hits = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            low = fn.lower()
            if low.endswith(".nii.gz") or low.endswith(".nii") or low.endswith(".h5"):
                hits.append(os.path.join(dirpath, fn))
    return hits


def main():
    parser = argparse.ArgumentParser(
        description="Zählt isotrop vs. anisotrop (falls beurteilbar) und gibt Größen-Statistiken aus für .nii(.gz) und fastMRI-.h5."
    )
    parser.add_argument("--rel-tol", type=float, default=1e-3,
                        help="Relative Toleranz für die Isotropie-Prüfung (default: 1e-3)")
    parser.add_argument("--abs-tol", type=float, default=1e-6,
                        help="Absolute Toleranz für die Isotropie-Prüfung (default: 1e-6)")
    parser.add_argument("--print-details", action="store_true",
                        help="Pro Datei den Status, Zooms/Spacings und Shape ausgeben")
    args = parser.parse_args()

    root = SEARCH_PATH

    files = find_files(root)
    total = len(files)
    if total == 0:
        print(f"Keine passenden Dateien gefunden unter: {root}")
        sys.exit(0)

    iso_true = 0
    iso_false = 0
    iso_na = 0
    error_count = 0
    shape_counter_3d: Counter[Tuple[int, ...]] = Counter()
    shape_counter_2d: Counter[Tuple[int, ...]] = Counter()
    count_nifti = 0
    count_h5 = 0

    for idx, path in enumerate(files, start=1):
        try:
            zooms, shp, typ = get_zooms_and_shape(path)
            if typ == "nifti":
                count_nifti += 1
                # NIfTI: shp sind die ersten 3 Dimensionen
                if len(shp) == 3:
                    shape_counter_3d[shp] += 1
            else:
                count_h5 += 1
                # fastMRI: shp ist 2D (H,W)
                if len(shp) == 2:
                    shape_counter_2d[shp] += 1

            iso = is_isotropic(zooms, rel_tol=args.rel_tol, abs_tol=args.abs_tol)
            if iso is True:
                iso_true += 1
            elif iso is False:
                iso_false += 1
            else:
                iso_na += 1

            if args.print_details:
                status = "isotrop" if iso is True else ("anisotrop" if iso is False else "N/A")
                # zoombeschreibung
                z_str = ", ".join("—" if z is None else f"{z:g}mm" for z in (zooms + (None,))[:3])
                print(f"[{status:9}] ({typ:10}) {path}\n"
                      f"           spacings(x,y,z): ({z_str})  |  shape: {shp}")

        except Exception as e:
            error_count += 1
            if args.print_details:
                print(f"[fehler   ] {path}  |  {e}")

        # Fortschrittsbalken
        percent = (idx / total) * 100
        bar_length = 30
        filled = int(bar_length * idx // total)
        bar = "█" * filled + "-" * (bar_length - filled)
        print(f"\rVerarbeite Dateien: |{bar}| {idx}/{total} ({percent:5.1f}%)", end="", flush=True)

    print()  # Umbruch

    print("\n--- Zusammenfassung ---")
    print(f"Gefundene Dateien gesamt     : {total}")
    print(f"  davon NIfTI (.nii/.nii.gz) : {count_nifti}")
    print(f"  davon fastMRI HDF5 (.h5)   : {count_h5}")
    print(f"Isotrop (beurteilbar)        : {iso_true}")
    print(f"Anisotrop (beurteilbar)      : {iso_false}")
    print(f"Isotropie N/A (z.B. 2D HDF5) : {iso_na}")
    print(f"Fehler beim Einlesen         : {error_count}")

    if shape_counter_3d:
        print("\n--- Häufigkeiten 3D-Bildgrößen (Pixel, NIfTI) ---")
        for shp, cnt in sorted(shape_counter_3d.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"{shp}  >  {cnt} mal")

    if shape_counter_2d:
        print("\n--- Häufigkeiten 2D-Matrixgrößen (Pixel, fastMRI .h5) ---")
        for shp, cnt in sorted(shape_counter_2d.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"{shp}  >  {cnt} mal")

    sys.exit(0 if error_count == 0 else 1)


if __name__ == "__main__":
    main()