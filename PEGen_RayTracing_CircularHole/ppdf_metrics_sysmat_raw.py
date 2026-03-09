import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import center_of_mass, label, map_coordinates


def unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("zero-length vector")
    return v / n


def normalize_volume(vol):
    vol = np.asarray(vol, dtype=np.float32)
    vol = np.maximum(vol, 0.0)
    vmax = float(vol.max())
    if vmax <= 0:
        return vol.copy()
    return vol / vmax


def connected_components(mask):
    structure = np.ones((3, 3, 3), dtype=np.int32)
    return label(mask.astype(bool), structure=structure)


def component_bbox(mask, spacing_zyx):
    idx = np.argwhere(mask)
    if idx.size == 0:
        return {
            "bbox_z_mm": 0.0,
            "bbox_y_mm": 0.0,
            "bbox_x_mm": 0.0,
            "active_volume_mm3": 0.0,
        }
    mins = idx.min(axis=0)
    maxs = idx.max(axis=0)
    spans_vox = (maxs - mins + 1).astype(float)
    spans_mm = spans_vox * np.asarray(spacing_zyx, dtype=float)
    active_volume_mm3 = float(idx.shape[0] * np.prod(spacing_zyx))
    return {
        "bbox_z_mm": float(spans_mm[0]),
        "bbox_y_mm": float(spans_mm[1]),
        "bbox_x_mm": float(spans_mm[2]),
        "active_volume_mm3": active_volume_mm3,
    }


def centroid_of_component(weighted_volume, mask):
    comp = np.asarray(weighted_volume, dtype=float) * mask.astype(float)
    if comp.sum() <= 0:
        idx = np.argwhere(mask)
        if idx.size == 0:
            raise ValueError("empty component")
        return idx.mean(axis=0)
    return np.asarray(center_of_mass(comp))


def extract_line_profile(volume, center_zyx, axis_zyx, spacing_zyx, half_length_mm=None, step_mm=None):
    volume = np.asarray(volume, dtype=float)
    center_zyx = np.asarray(center_zyx, dtype=float)
    axis_zyx = unit(axis_zyx)
    spacing_zyx = np.asarray(spacing_zyx, dtype=float)

    if half_length_mm is None:
        diag_mm = np.linalg.norm(np.asarray(volume.shape) * spacing_zyx)
        half_length_mm = 0.6 * diag_mm
    if step_mm is None:
        step_mm = float(np.min(spacing_zyx) / 2.0)

    s_mm = np.arange(-half_length_mm, half_length_mm + step_mm, step_mm)
    delta_vox = (s_mm[:, None] * axis_zyx[None, :]) / spacing_zyx[None, :]
    coords = center_zyx[None, :] + delta_vox
    coords = coords.T
    profile = map_coordinates(volume, coords, order=1, mode="constant", cval=0.0)
    return s_mm, profile


def crossing_position(x, y, level, side):
    if side == "left":
        for i in range(len(y) - 1):
            y0, y1 = y[i], y[i + 1]
            if (y0 <= level <= y1) or (y1 <= level <= y0):
                if y1 == y0:
                    return float(x[i])
                t = (level - y0) / (y1 - y0)
                return float(x[i] + t * (x[i + 1] - x[i]))
    elif side == "right":
        for i in range(len(y) - 1):
            y0, y1 = y[i], y[i + 1]
            if (y0 >= level >= y1) or (y1 >= level >= y0):
                if y1 == y0:
                    return float(x[i])
                t = (level - y0) / (y1 - y0)
                return float(x[i] + t * (x[i + 1] - x[i]))
    return None


def compute_fwhm_and_edge_sharpness(s_mm, profile):
    profile = np.asarray(profile, dtype=float)
    s_mm = np.asarray(s_mm, dtype=float)

    if profile.size == 0 or profile.max() <= 0:
        return {
            "fwhm_mm": np.nan,
            "edge_left_25_75_mm": np.nan,
            "edge_right_25_75_mm": np.nan,
            "peak_mm": np.nan,
        }

    p = profile / profile.max()
    peak_idx = int(np.argmax(p))
    peak_x = float(s_mm[peak_idx])

    left_x = s_mm[:peak_idx + 1]
    left_p = p[:peak_idx + 1]
    right_x = s_mm[peak_idx:]
    right_p = p[peak_idx:]

    x50_left = crossing_position(left_x, left_p, 0.5, "left")
    x50_right = crossing_position(right_x, right_p, 0.5, "right")
    fwhm_mm = np.nan
    if x50_left is not None and x50_right is not None:
        fwhm_mm = float(x50_right - x50_left)

    x25_left = crossing_position(left_x, left_p, 0.25, "left")
    x75_left = crossing_position(left_x, left_p, 0.75, "left")
    edge_left = np.nan
    if x25_left is not None and x75_left is not None:
        edge_left = float(abs(x75_left - x25_left))

    x75_right = crossing_position(right_x, right_p, 0.75, "right")
    x25_right = crossing_position(right_x, right_p, 0.25, "right")
    edge_right = np.nan
    if x75_right is not None and x25_right is not None:
        edge_right = float(abs(x25_right - x75_right))

    return {
        "fwhm_mm": fwhm_mm,
        "edge_left_25_75_mm": edge_left,
        "edge_right_25_75_mm": edge_right,
        "peak_mm": peak_x,
    }


def choose_components(volume, support_threshold_fraction=0.10, min_component_voxels=5):
    vol = normalize_volume(volume)
    mask = vol >= support_threshold_fraction
    labels, n = connected_components(mask)

    kept = np.zeros_like(labels)
    new_id = 0
    for cid in range(1, n + 1):
        comp = labels == cid
        if int(comp.sum()) >= min_component_voxels:
            new_id += 1
            kept[comp] = new_id
    return kept, new_id


def analyze_single_component(volume, component_mask, spacing_zyx, analysis_axis_zyx):
    center = centroid_of_component(volume, component_mask)
    masked_volume = volume * component_mask.astype(float)
    s_mm, profile = extract_line_profile(
        masked_volume,
        center_zyx=center,
        axis_zyx=analysis_axis_zyx,
        spacing_zyx=spacing_zyx,
    )
    metrics = compute_fwhm_and_edge_sharpness(s_mm, profile)
    metrics.update(component_bbox(component_mask, spacing_zyx))
    metrics["centroid_z_vox"] = float(center[0])
    metrics["centroid_y_vox"] = float(center[1])
    metrics["centroid_x_vox"] = float(center[2])
    return metrics, s_mm, profile


def save_profile_png(s_mm, profile, png_path, title):
    p = np.asarray(profile, dtype=float)
    if p.max() > 0:
        p = p / p.max()

    fig = plt.figure(figsize=(7, 4.5))
    ax = fig.add_subplot(111)
    ax.plot(s_mm, p, linewidth=2)
    ax.axhline(0.5, linestyle="--", linewidth=1)
    ax.axhline(0.25, linestyle=":", linewidth=1)
    ax.axhline(0.75, linestyle=":", linewidth=1)
    ax.set_xlabel("Position along analysis axis (mm)")
    ax.set_ylabel("Normalized profile")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(png_path, dpi=180)
    plt.close(fig)


def save_3d_png(volume, component_mask, spacing_zyx, analysis_axis_zyx, png_path, title, render_threshold_fraction=0.10):
    vol = np.asarray(volume, dtype=float)
    spacing_zyx = np.asarray(spacing_zyx, dtype=float)
    masked = vol * component_mask.astype(float)
    vmax = masked.max()
    if vmax <= 0:
        return

    keep = masked >= (render_threshold_fraction * vmax)
    pts = np.argwhere(keep)
    vals = masked[keep]
    if pts.shape[0] == 0:
        return

    xyz = pts[:, ::-1].astype(float)
    xyz[:, 0] *= spacing_zyx[2]
    xyz[:, 1] *= spacing_zyx[1]
    xyz[:, 2] *= spacing_zyx[0]

    center = centroid_of_component(masked, component_mask)
    center_xyz = np.array([
        center[2] * spacing_zyx[2],
        center[1] * spacing_zyx[1],
        center[0] * spacing_zyx[0],
    ])

    axis = unit(np.asarray(analysis_axis_zyx, dtype=float))
    axis_xyz = np.array([
        axis[2] * spacing_zyx[2],
        axis[1] * spacing_zyx[1],
        axis[0] * spacing_zyx[0],
    ])
    axis_xyz = unit(axis_xyz)

    line_half = 0.25 * np.linalg.norm(np.asarray(vol.shape) * spacing_zyx)
    line = np.vstack([center_xyz - line_half * axis_xyz, center_xyz + line_half * axis_xyz])

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=vals, s=8, alpha=0.45)
    ax.plot(line[:, 0], line[:, 1], line[:, 2], linewidth=2)
    ax.scatter([center_xyz[0]], [center_xyz[1]], [center_xyz[2]], s=50, marker="x")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("z (mm)")
    ax.set_title(title)
    fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.08, label="PPDF intensity")
    fig.tight_layout()
    fig.savefig(png_path, dpi=180)
    plt.close(fig)


def save_population_histograms(metrics, png_path):
    if not metrics:
        return

    fwhm = np.array([m["fwhm_mm"] for m in metrics], dtype=float)
    edge = np.array([
        np.nanmean([m["edge_left_25_75_mm"], m["edge_right_25_75_mm"]])
        for m in metrics
    ], dtype=float)
    active = np.array([m["active_volume_mm3"] for m in metrics], dtype=float)

    fig = plt.figure(figsize=(12, 3.8))
    axes = [fig.add_subplot(1, 3, i + 1) for i in range(3)]

    axes[0].hist(fwhm[np.isfinite(fwhm)], bins=24)
    axes[0].set_title("FWHM distribution")
    axes[0].set_xlabel("mm")

    axes[1].hist(edge[np.isfinite(edge)], bins=24)
    axes[1].set_title("Edge 25-75 distribution")
    axes[1].set_xlabel("mm")

    axes[2].hist(active[np.isfinite(active)], bins=24)
    axes[2].set_title("Active volume distribution")
    axes[2].set_xlabel("mm^3")

    for ax in axes:
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(png_path, dpi=180)
    plt.close(fig)


def open_sysmat_row(sysmat_path, num_det, nz, ny, nx, row_idx, dtype=np.float32):
    vox = nz * ny * nx
    expected_bytes = num_det * vox * np.dtype(dtype).itemsize
    actual_bytes = os.path.getsize(sysmat_path)

    if actual_bytes != expected_bytes:
        raise RuntimeError(
            f"Size mismatch:\n"
            f"  file bytes    = {actual_bytes}\n"
            f"  expected bytes= {expected_bytes}\n"
            f"  num_det={num_det}, nz={nz}, ny={ny}, nx={nx}, vox={vox}, dtype={dtype}"
        )

    mm = np.memmap(sysmat_path, dtype=dtype, mode="r", shape=(num_det, vox))
    row = np.asarray(mm[row_idx], dtype=np.float32).reshape((nz, ny, nx), order="C")
    return row


def main():
    ap = argparse.ArgumentParser(description="PPDF metrics for raw float32 .sysmat")
    ap.add_argument("--sysmat", required=True, help="Raw float32 .sysmat path")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--num_det", type=int, required=True)
    ap.add_argument("--nz", type=int, required=True)
    ap.add_argument("--ny", type=int, required=True)
    ap.add_argument("--nx", type=int, required=True)
    ap.add_argument("--rows", nargs="+", type=int, default=[0], help="Detector rows to analyze. Default: 0")
    ap.add_argument("--voxel_size_mm", nargs=3, type=float, required=True, metavar=("Z", "Y", "X"))
    ap.add_argument("--analysis_axis_zyx", nargs=3, type=float, default=[0.0, 0.0, 1.0], metavar=("AZ", "AY", "AX"))
    ap.add_argument("--support_threshold_fraction", type=float, default=0.10)
    ap.add_argument("--min_component_voxels", type=int, default=5)
    ap.add_argument("--render_threshold_fraction", type=float, default=0.10)
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    spacing = np.asarray(args.voxel_size_mm, dtype=float)
    axis = unit(np.asarray(args.analysis_axis_zyx, dtype=float))

    all_metrics = []

    for row_idx in args.rows:
        vol = open_sysmat_row(
            args.sysmat,
            num_det=args.num_det,
            nz=args.nz,
            ny=args.ny,
            nx=args.nx,
            row_idx=row_idx,
        )

        vol = normalize_volume(vol)
        labels, ncomp = choose_components(
            vol,
            support_threshold_fraction=args.support_threshold_fraction,
            min_component_voxels=args.min_component_voxels,
        )

        if ncomp == 0:
            print(f"[WARN] row {row_idx}: no active component found")
            continue

        for cid in range(1, ncomp + 1):
            comp_mask = labels == cid
            metrics, s_mm, profile = analyze_single_component(
                vol,
                comp_mask,
                spacing_zyx=spacing,
                analysis_axis_zyx=axis,
            )
            metrics["row_idx"] = int(row_idx)
            metrics["component_idx"] = int(cid)
            all_metrics.append(metrics)

            base = output_dir / f"row_{row_idx:04d}_comp_{cid:02d}"

            save_3d_png(
                vol,
                comp_mask,
                spacing_zyx=spacing,
                analysis_axis_zyx=axis,
                png_path=base.with_name(base.name + "_3d.png"),
                title=f"Row {row_idx} | Component {cid}",
                render_threshold_fraction=args.render_threshold_fraction,
            )

            save_profile_png(
                s_mm,
                profile,
                png_path=base.with_name(base.name + "_profile.png"),
                title=f"Row {row_idx} | Component {cid} profile",
            )

            print(
                f"[OK] row={row_idx} comp={cid} "
                f"FWHM={metrics['fwhm_mm']:.4f} mm "
                f"edgeL={metrics['edge_left_25_75_mm']:.4f} mm "
                f"edgeR={metrics['edge_right_25_75_mm']:.4f} mm "
                f"activeVol={metrics['active_volume_mm3']:.4f} mm^3"
            )

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    save_population_histograms(all_metrics, output_dir / "population_histograms.png")
    print(f"[DONE] wrote results to {output_dir}")


if __name__ == "__main__":
    main()
