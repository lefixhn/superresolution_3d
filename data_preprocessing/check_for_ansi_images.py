'''
This script searches through all .nii.gz files in a given directory. 
It checks whether all images are isotropic, or whether there are also anisotropic images.
Now includes shape statistics and a simple progress bar.
This code is AI generated
'''
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import math
from typing import Tuple, List
from collections import Counter

SEARCH_PATH = "/content/drive/MyDrive/superresolution_3d_data/datasets/BraTS2021_Training_Data"
print("analyzing data")
try:
    import nibabel as nib
except ImportError:
    print("Fehler: 'nibabel' ist nicht installiert. Installiere es mit: pip install nibabel")
    sys.exit(1)


def is_isotropic(zooms: Tuple[float, ...], rel_tol: float = 1e-3, abs_tol: float = 1e-6) -> bool:
    if len(zooms) < 3:
        return False
    z0, z1, z2 = zooms[:3]
    return (math.isclose(z0, z1, rel_tol=rel_tol, abs_tol=abs_tol) and
            math.isclose(z1, z2, rel_tol=rel_tol, abs_tol=abs_tol))


def get_zooms(path: str) -> Tuple[float, ...]:
    img = nib.load(path)
    return img.header.get_zooms()


def get_spatial_shape(path: str) -> Tuple[int, ...]:
    img = nib.load(path)
    shape = img.shape
    if len(shape) >= 3:
        return tuple(shape[:3])
    else:
        return tuple(shape)


def find_files(root: str) -> List[str]:
    hits = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith(".nii.gz"):
                hits.append(os.path.join(dirpath, fn))
    return hits


def main():
    parser = argparse.ArgumentParser(
        description="Zählt isotrope vs. anisotrope NIfTI-Dateien (.nii.gz) und gibt eine Statistik der Bildgrößen aus."
    )
    parser.add_argument("--rel-tol", type=float, default=1e-3,
                        help="Relative Toleranz für die Isotropie-Prüfung (default: 1e-3)")
    parser.add_argument("--abs-tol", type=float, default=1e-6,
                        help="Absolute Toleranz für die Isotropie-Prüfung (default: 1e-6)")
    parser.add_argument("--print-details", action="store_true",
                        help="Pro Datei den Status und die Zooms ausgeben")
    args = parser.parse_args()

    root = SEARCH_PATH

    files = find_files(root)
    total = len(files)
    iso_count = 0
    aniso_count = 0
    error_count = 0
    shape_counter: Counter[Tuple[int, ...]] = Counter()

    for idx, path in enumerate(files, start=1):
        try:
            zooms = get_zooms(path)
            iso = is_isotropic(zooms, rel_tol=args.rel_tol, abs_tol=args.abs_tol)
            if iso:
                iso_count += 1
            else:
                aniso_count += 1

            shp = get_spatial_shape(path)
            shape_counter[shp] += 1

            if args.print_details:
                status = "isotrop" if iso else "anisotrop"
                z_str = ", ".join(f"{z:g}" for z in zooms[:4])
                print(f"[{status:9}] {path}  |  zooms: ({z_str}{', ...' if len(zooms) > 4 else ''})  |  shape: {shp}")

        except Exception as e:
            error_count += 1
            if args.print_details:
                print(f"[fehler   ] {path}  |  {e}")

        # Fortschrittsanzeige nur alle 10 Dateien aktualisieren
        if idx % 10 == 0 or idx == total:
            percent = (idx / total) * 100
            bar_length = 30
            filled = int(bar_length * idx // total)
            bar = "█" * filled + "-" * (bar_length - filled)
            print(f"\rVerarbeite Dateien: |{bar}| {idx}/{total} ({percent:5.1f}%)", end="", flush=True)

    print()  # Umbruch nach Fortschrittsbalken

    print("\n--- Zusammenfassung ---")
    print(f"Gefundene .nii.gz-Dateien : {total}")
    print(f"Isotrop                    : {iso_count}")
    print(f"Anisotrop                  : {aniso_count}")
    print(f"Fehler beim Einlesen       : {error_count}")

    if shape_counter:
        print("\n--- Häufigkeiten der Bildgrößen (Pixel) ---")
        for shp, cnt in sorted(shape_counter.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"{shp}  >  {cnt} mal enthalten")

    sys.exit(0 if error_count == 0 else 1)


if __name__ == "__main__":
    main()