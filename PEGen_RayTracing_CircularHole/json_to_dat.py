#!/usr/bin/env python3
"""
json_to_dat.py

Convert annotated geometry JSON into the 4 binary float32 .dat files expected by:
GPU-Based-System-Matrix-Calculation-for-SPECT-PET

Key points (based on repo docs + your sanity script behavior):
- Param_Image.dat   : 12 float32, fixed indices 0..11
- Param_Physics.dat : 10 float32, fixed indices 0..9
- Param_Detector.dat: 1 + 12*numBins float32
- Param_Collimator.dat:
    col[0] = numLayers
    layer table is 10 float32 per layer, STARTING at col[layer_base] (default 10)
    holes table is 9 float32 per hole, STARTING at col[holes_base] (default 100)

This default (layer_base=10) is consistent with check_geometry_sanity.py showing
Layer0 fields as 0 when the layer record was written at index 1.
"""

import argparse
import json
import os
from typing import Any, Dict, List

import numpy as np

F32 = np.dtype("<f4")  # explicit little-endian float32


def die(msg: str) -> None:
    raise SystemExit(f"ERROR: {msg}")


def require(cond: bool, msg: str) -> None:
    if not cond:
        die(msg)


def get(d: Dict[str, Any], key: str, *, where: str) -> Any:
    if key not in d:
        die(f"Missing key '{key}' in {where}")
    return d[key]


def as_float(x: Any, where: str) -> float:
    try:
        return float(x)
    except Exception:
        die(f"Expected a number at {where}, got: {x!r}")


def as_int(x: Any, where: str) -> int:
    try:
        return int(x)
    except Exception:
        die(f"Expected an int at {where}, got: {x!r}")


def write_f32(path: str, arr: np.ndarray) -> None:
    arr = np.asarray(arr, dtype=F32)
    arr.tofile(path)


def read_f32(path: str) -> np.ndarray:
    return np.fromfile(path, dtype=F32)


def build_param_image(image: Dict[str, Any]) -> np.ndarray:
    # Param_Image.dat = 12 float32 values, fixed indices
    out = np.zeros(12, dtype=F32)

    out[0] = as_int(get(image, "num_voxel_x", where="image"), "image.num_voxel_x")
    out[1] = as_int(get(image, "num_voxel_y", where="image"), "image.num_voxel_y")
    out[2] = as_int(get(image, "num_voxel_z", where="image"), "image.num_voxel_z")

    out[3] = as_float(get(image, "voxel_size_x_mm", where="image"), "image.voxel_size_x_mm")
    out[4] = as_float(get(image, "voxel_size_y_mm", where="image"), "image.voxel_size_y_mm")
    out[5] = as_float(get(image, "voxel_size_z_mm", where="image"), "image.voxel_size_z_mm")

    out[6] = as_int(get(image, "num_rotation", where="image"), "image.num_rotation")
    out[7] = as_float(get(image, "angle_per_rotation_rad", where="image"), "image.angle_per_rotation_rad")

    out[8] = as_float(get(image, "shift_fov_x_mm", where="image"), "image.shift_fov_x_mm")
    out[9] = as_float(get(image, "shift_fov_y_mm", where="image"), "image.shift_fov_y_mm")
    out[10] = as_float(get(image, "shift_fov_z_mm", where="image"), "image.shift_fov_z_mm")

    out[11] = as_float(get(image, "fov_to_collimator0_mm", where="image"), "image.fov_to_collimator0_mm")

    return out


def build_param_physics(phys: Dict[str, Any]) -> np.ndarray:
    # Param_Physics.dat = 10 float32 values, fixed indices
    out = np.zeros(10, dtype=F32)

    out[0] = as_float(get(phys, "use_compton", where="physics"), "physics.use_compton")
    out[1] = as_float(get(phys, "save_pe_sysmat", where="physics"), "physics.save_pe_sysmat")
    out[2] = as_float(get(phys, "save_compton_sysmat", where="physics"), "physics.save_compton_sysmat")
    out[3] = as_float(get(phys, "save_pecompton_sysmat", where="physics"), "physics.save_pecompton_sysmat")

    out[4] = as_float(get(phys, "use_same_energy_window", where="physics"), "physics.use_same_energy_window")
    out[5] = as_float(get(phys, "energy_window_lower", where="physics"), "physics.energy_window_lower")
    out[6] = as_float(get(phys, "energy_window_upper", where="physics"), "physics.energy_window_upper")
    out[7] = as_float(get(phys, "target_pe_energy", where="physics"), "physics.target_pe_energy")

    out[8] = as_float(get(phys, "calc_crystal_geometry_relationship", where="physics"),
                      "physics.calc_crystal_geometry_relationship")
    out[9] = as_float(get(phys, "calc_collimator_geometry_relationship", where="physics"),
                      "physics.calc_collimator_geometry_relationship")

    return out


def build_param_detector(det: Dict[str, Any]) -> np.ndarray:
    bins = get(det, "bins", where="detector")
    require(isinstance(bins, list) and len(bins) > 0, "detector.bins must be a non-empty list")

    n = len(bins)
    # Param_Detector.dat = [numBins] + 12 floats per bin
    out = np.zeros(1 + 12 * n, dtype=F32)
    out[0] = float(n)

    for i, b in enumerate(bins):
        where = f"detector.bins[{i}]"
        require(isinstance(b, dict), f"{where} must be an object")

        base = 1 + i * 12
        out[base + 0] = as_float(get(b, "cx", where=where), f"{where}.cx")
        out[base + 1] = as_float(get(b, "cy", where=where), f"{where}.cy")
        out[base + 2] = as_float(get(b, "cz", where=where), f"{where}.cz")

        out[base + 3] = as_float(get(b, "width_mm", where=where), f"{where}.width_mm")
        out[base + 4] = as_float(get(b, "thickness_mm", where=where), f"{where}.thickness_mm")
        out[base + 5] = as_float(get(b, "height_mm", where=where), f"{where}.height_mm")

        out[base + 6] = as_float(get(b, "mu_total", where=where), f"{where}.mu_total")
        out[base + 7] = as_float(get(b, "mu_pe", where=where), f"{where}.mu_pe")
        out[base + 8] = as_float(get(b, "mu_compton", where=where), f"{where}.mu_compton")

        out[base + 9] = as_float(get(b, "energy_resolution_at_target", where=where),
                                 f"{where}.energy_resolution_at_target")
        out[base + 10] = as_float(get(b, "rotation_about_y_rad", where=where),
                                  f"{where}.rotation_about_y_rad")
        out[base + 11] = as_float(get(b, "flag", where=where), f"{where}.flag")

    return out


def flatten_collimator_holes(layers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    holes_flat: List[Dict[str, Any]] = []
    for li, layer in enumerate(layers):
        where = f"collimator.layers[{li}]"
        hc = as_int(get(layer, "hole_count", where=where), f"{where}.hole_count")
        holes = get(layer, "holes", where=where)
        require(isinstance(holes, list), f"{where}.holes must be a list")
        require(len(holes) == hc, f"{where}: hole_count={hc} but holes has {len(holes)} entries")

        for hi, h in enumerate(holes):
            require(isinstance(h, dict), f"{where}.holes[{hi}] must be an object")
            holes_flat.append(h)

    return holes_flat


def build_param_collimator(col: Dict[str, Any], layer_base: int, holes_base: int) -> np.ndarray:
    layers = get(col, "layers", where="collimator")
    require(isinstance(layers, list) and len(layers) > 0, "collimator.layers must be a non-empty list")

    require(layer_base >= 0, "layer_base must be >= 0")
    require(holes_base >= 0, "holes_base must be >= 0")
    require(holes_base >= layer_base, "holes_base should be >= layer_base")

    num_layers = len(layers)
    holes_flat = flatten_collimator_holes(layers)
    num_holes_total = len(holes_flat)

    # Layer table = 10 floats per layer, starting at layer_base
    layer_table_len = layer_base + 10 * num_layers
    # Hole table = 9 floats per hole, starting at holes_base
    hole_table_len = holes_base + 9 * num_holes_total

    out_len = max(layer_table_len, hole_table_len, 1)
    out = np.zeros(out_len, dtype=F32)

    # Global
    out[0] = float(num_layers)

    # Per-layer table
    for li, layer in enumerate(layers):
        where = f"collimator.layers[{li}]"
        base = layer_base + li * 10

        out[base + 0] = float(as_int(get(layer, "hole_count", where=where), f"{where}.hole_count"))
        out[base + 1] = as_float(get(layer, "width_mm", where=where), f"{where}.width_mm")
        out[base + 2] = as_float(get(layer, "thickness_mm", where=where), f"{where}.thickness_mm")
        out[base + 3] = as_float(get(layer, "height_mm", where=where), f"{where}.height_mm")
        out[base + 4] = as_float(get(layer, "offset_from_layer0_mm", where=where),
                                 f"{where}.offset_from_layer0_mm")

        out[base + 5] = as_float(get(layer, "mu_total", where=where), f"{where}.mu_total")
        out[base + 6] = as_float(get(layer, "mu_pe", where=where), f"{where}.mu_pe")
        out[base + 7] = as_float(get(layer, "mu_compton", where=where), f"{where}.mu_compton")

        # base+8 and base+9: not defined in README -> left as 0.0

    # Per-hole table (flattened order), starting at holes_base
    for hi, h in enumerate(holes_flat):
        where = f"collimator.holes_flat[{hi}]"
        base = holes_base + hi * 9

        out[base + 0] = as_float(get(h, "cx", where=where), f"{where}.cx")
        out[base + 1] = as_float(get(h, "cy1", where=where), f"{where}.cy1")
        out[base + 2] = as_float(get(h, "cy2", where=where), f"{where}.cy2")
        out[base + 3] = as_float(get(h, "cz", where=where), f"{where}.cz")
        out[base + 4] = as_float(get(h, "radius", where=where), f"{where}.radius")

        out[base + 5] = as_float(get(h, "mu_total", where=where), f"{where}.mu_total")
        out[base + 6] = as_float(get(h, "mu_pe", where=where), f"{where}.mu_pe")
        out[base + 7] = as_float(get(h, "mu_compton", where=where), f"{where}.mu_compton")
        out[base + 8] = as_float(get(h, "flag", where=where), f"{where}.flag")

    return out


def print_quick_verify(outdir: str, layer_base: int, holes_base: int) -> None:
    col = read_f32(os.path.join(outdir, "Param_Collimator.dat"))
    det = read_f32(os.path.join(outdir, "Param_Detector.dat"))
    img = read_f32(os.path.join(outdir, "Param_Image.dat"))
    phy = read_f32(os.path.join(outdir, "Param_Physics.dat"))

    print("\n=== QUICK VERIFY (read-back) ===")
    print(f"Param_Image:   floats={img.size} head={img[:12].tolist()}")
    print(f"Param_Physics: floats={phy.size} head={phy[:10].tolist()}")
    print(f"Param_Detector:floats={det.size} numBins={det[0]} last(flag)={det[-1]}")
    print(f"Param_Collimator:floats={col.size} numLayers={col[0]}")
    if col.size >= layer_base + 8:
        print(f"  Layer0 @ base={layer_base}: "
              f"numHoles={col[layer_base+0]} widthX={col[layer_base+1]} thickY={col[layer_base+2]} "
              f"heightZ={col[layer_base+3]} dist={col[layer_base+4]} "
              f"muTot={col[layer_base+5]} muPE={col[layer_base+6]} muC={col[layer_base+7]}")
    if col.size >= holes_base + 9:
        print(f"  First hole @ base={holes_base}: {col[holes_base:holes_base+9].tolist()}")
    print("=== END VERIFY ===\n")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Convert annotated geometry JSON into binary float32 Param_*.dat files."
    )
    ap.add_argument("json_path", help="Path to geometry JSON (can include _comment_* keys; ignored).")
    ap.add_argument("--outdir", default=".", help="Directory to write Param_*.dat files (default: .)")

    ap.add_argument(
        "--collimator-layer-base",
        type=int,
        default=10,
        help=(
            "Float index where the 10-float-per-layer table starts in Param_Collimator.dat. "
            "Default=10 matches repo sanity tooling behavior."
        ),
    )
    ap.add_argument(
        "--collimator-holes-base",
        type=int,
        default=100,
        help="Float index where the 9-float-per-hole table starts in Param_Collimator.dat. Default=100.",
    )

    ap.add_argument("--print-summary", action="store_true", help="Print sizes/summary after writing.")
    ap.add_argument("--verify", action="store_true", help="Read back the written .dat files and print key fields.")
    args = ap.parse_args()

    require(args.collimator_layer_base >= 0, "--collimator-layer-base must be >= 0")
    require(args.collimator_holes_base >= 0, "--collimator-holes-base must be >= 0")

    with open(args.json_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    col = get(cfg, "collimator", where="root")
    det = get(cfg, "detector", where="root")
    img = get(cfg, "image", where="root")
    phy = get(cfg, "physics", where="root")

    os.makedirs(args.outdir, exist_ok=True)

    param_coll = build_param_collimator(col, layer_base=args.collimator_layer_base, holes_base=args.collimator_holes_base)
    param_det = build_param_detector(det)
    param_img = build_param_image(img)
    param_phy = build_param_physics(phy)

    write_f32(os.path.join(args.outdir, "Param_Collimator.dat"), param_coll)
    write_f32(os.path.join(args.outdir, "Param_Detector.dat"), param_det)
    write_f32(os.path.join(args.outdir, "Param_Image.dat"), param_img)
    write_f32(os.path.join(args.outdir, "Param_Physics.dat"), param_phy)

    if args.print_summary:
        layers = len(get(col, "layers", where="collimator"))
        holes = sum(int(L["hole_count"]) for L in col["layers"])
        bins = len(get(det, "bins", where="detector"))
        print("Wrote:")
        print(f"  Param_Collimator.dat : {param_coll.size} float32 ({param_coll.nbytes} bytes) | layers={layers}, holes={holes}"
              f" | layer_base={args.collimator_layer_base} holes_base={args.collimator_holes_base}")
        print(f"  Param_Detector.dat   : {param_det.size} float32 ({param_det.nbytes} bytes) | bins={bins}")
        print(f"  Param_Image.dat      : {param_img.size} float32 ({param_img.nbytes} bytes)")
        print(f"  Param_Physics.dat    : {param_phy.size} float32 ({param_phy.nbytes} bytes)")

    if args.verify:
        print_quick_verify(args.outdir, args.collimator_layer_base, args.collimator_holes_base)


if __name__ == "__main__":
    main()
