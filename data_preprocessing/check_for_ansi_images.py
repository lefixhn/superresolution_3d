'''
This script searches through all .nii.gz files in a given directory. 
It checks weather all images are isotropic, or weather there are also ansitropic images
This code is AI generated
'''
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import math
from typing import Tuple, List, Dict
from collections import Counter

SEARCH_PATH = "/content/drive/MyDrive/superresolution_3d_data/datasets/BraTS2021_Training_Data"

try:
    import nibabel as nib
except ImportError:
    print("Fehler: 'nibabel' ist nicht installiert. Installiere es mit: pip install nibabel")
    sys.exit(1)


def is_isotropic(zooms: Tuple[float, ...], rel_tol: float = 1e-3, abs_tol: float = 1e-6) -> bool:
    """
    Prüft, ob die ersten drei Zooms (Voxelgrößen) isotrop sind.
    Nutzt eine Kombination aus relativer und absoluter Toleranz.
    """
    if len(zooms) < 3:
        # Für 2D o.ä. behandeln wir das als nicht eindeutig → anisotrop
        return False
    z0, z1, z2 = zooms[:3]
    return (math.isclose(z0, z1, rel_tol=rel_tol, abs_tol=abs_tol) and
            math.isclose(z1, z2, rel_tol=rel_tol, abs_tol=abs_tol))


def get_zooms(path: str) -> Tuple[float, ...]:
    """
    Lädt die Datei mit nibabel und gibt die Zooms/Spacings zurück.
    """
    img = nib.load(path)
    # header.get_zooms liefert Voxelgrößen (z.B. (sx, sy, sz, t, ...))
    return img.header.get_zooms()


def get_spatial_shape(path: str) -> Tuple[int, ...]:
    """
    Liefert die ersten bis zu drei Dimensionen der Bildgröße (in Pixeln).
    Lädt nur den Header (kein volles Bild in RAM).
    Beispiele:
      - 3D Volumen (D, H, W) -> (D, H, W)
      - 2D Bild (H, W)       -> (H, W)
      - 4D (D, H, W, T)      -> (D, H, W)
    """
    img = nib.load(path)
    shape = img.shape  # Tuple[int, ...]
    if len(shape) >= 3:
        return tuple(shape[:3])
    else:
        return tuple(shape)  # z.B. (H, W)


def find_files(root: str) -> List[str]:
    """
    Sucht rekursiv nach .nii.gz-Dateien.
    """
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

    # Häufigkeiten der Shapes (erste drei Dimensionen)
    shape_counter: Counter[Tuple[int, ...]] = Counter()

    for path in files:
        try:
            zooms = get_zooms(path)
            iso = is_isotropic(zooms, rel_tol=args.rel_tol, abs_tol=args.abs_tol)
            if iso:
                iso_count += 1
            else:
                aniso_count += 1

            # Shape (Pixel) mitzählen
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

    print("\n--- Zusammenfassung ---")
    print(f"Gefundene .nii.gz-Dateien : {total}")
    print(f"Isotrop                    : {iso_count}")
    print(f"Anisotrop                  : {aniso_count}")
    print(f"Fehler beim Einlesen       : {error_count}")

    # Statistik der Bildgrößen ausgeben (absteigend nach Häufigkeit)
    if shape_counter:
        print("\n--- Häufigkeiten der Bildgrößen (Pixel) ---")
        for shp, cnt in sorted(shape_counter.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"{shp}  >  {cnt} mal enthalten")

    # Exit-Code: 0 ok, 1 wenn Fehler auftraten
    sys.exit(0 if error_count == 0 else 1)


if __name__ == "__main__":
    main()