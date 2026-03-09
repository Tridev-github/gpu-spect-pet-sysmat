import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import center_of_mass, label, map_coordinates


ArrayLike = np.ndarray


def unit_vector(v: Sequence[float]) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError('Zero-length vector is not allowed.')
    return v / n


def voxel_spacing_zyx(voxel_size_mm: Sequence[float]) -> np.ndarray:
    if len(voxel_size_mm) != 3:
        raise ValueError('voxel_size_mm must have 3 values: z y x')
    return np.asarray(voxel_size_mm, dtype=float)


def normalize_volume(vol: ArrayLike) -> ArrayLike:
    vol = np.asarray(vol, dtype=float)
    vol = np.maximum(vol, 0.0)
    vmax = vol.max()
    if vmax <= 0:
        return vol.copy()
    return vol / vmax


def connected_components(mask: ArrayLike) -> Tuple[ArrayLike, int]:
    structure = np.ones((3, 3, 3), dtype=int)
    return label(mask.astype(bool), structure=structure)


def component_bbox(mask: ArrayLike, spacing_zyx: np.ndarray) -> Dict[str, float]:
    idx = np.argwhere(mask)
    if idx.size == 0:
        return {
            'bbox_z_mm': 0.0,
            'bbox_y_mm': 0.0,
            'bbox_x_mm': 0.0,
            'active_volume_mm3': 0.0,
        }
    mins = idx.min(axis=0)
    maxs = idx.max(axis=0)
    spans_vox = (maxs - mins + 1).astype(float)
    spans_mm = spans_vox * spacing_zyx
    active_volume_mm3 = float(idx.shape[0] * np.prod(spacing_zyx))
    return {
        'bbox_z_mm': float(spans_mm[0]),
        'bbox_y_mm': float(spans_mm[1]),
        'bbox_x_mm': float(spans_mm[2]),
        'active_volume_mm3': active_volume_mm3,
    }


def make_orthonormal_basis(axis_zyx: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = unit_vector(axis_zyx)
    tmp = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(a, tmp)) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0])
    b = np.cross(a, tmp)
    b = unit_vector(b)
    c = np.cross(a, b)
    c = unit_vector(c)
    return a, b, c


def extract_line_profile(
    volume: ArrayLike,
    center_zyx: Sequence[float],
    axis_zyx: Sequence[float],
    spacing_zyx: Sequence[float],
    half_length_mm: Optional[float] = None,
    step_mm: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    volume = np.asarray(volume, dtype=float)
    center_zyx = np.asarray(center_zyx, dtype=float)
    axis_zyx = unit_vector(axis_zyx)
    spacing_zyx = np.asarray(spacing_zyx, dtype=float)

    if half_length_mm is None:
        # Conservative size: half of the volume diagonal in mm.
        diag_mm = np.linalg.norm(np.asarray(volume.shape) * spacing_zyx)
        half_length_mm = 0.6 * diag_mm
    if step_mm is None:
        step_mm = float(np.min(spacing_zyx) / 2.0)

    s_mm = np.arange(-half_length_mm, half_length_mm + step_mm, step_mm)
    delta_vox = (s_mm[:, None] * axis_zyx[None, :]) / spacing_zyx[None, :]
    coords = center_zyx[None, :] + delta_vox
    coords = coords.T  # shape: (3, n)
    profile = map_coordinates(volume, coords, order=1, mode='constant', cval=0.0)
    return s_mm, profile


def first_crossing_position(x: np.ndarray, y: np.ndarray, level: float, side: str) -> Optional[float]:
    if side not in {'left', 'right'}:
        raise ValueError("side must be 'left' or 'right'")
    if len(x) != len(y):
        raise ValueError('x and y lengths differ')

    if side == 'left':
        for i in range(len(y) - 1):
            y0, y1 = y[i], y[i + 1]
            if (y0 <= level <= y1) or (y1 <= level <= y0):
                if y1 == y0:
                    return float(x[i])
                t = (level - y0) / (y1 - y0)
                return float(x[i] + t * (x[i + 1] - x[i]))
    else:
        for i in range(len(y) - 1):
            y0, y1 = y[i], y[i + 1]
            if (y0 >= level >= y1) or (y1 >= level >= y0):
                if y1 == y0:
                    return float(x[i])
                t = (level - y0) / (y1 - y0)
                return float(x[i] + t * (x[i + 1] - x[i]))
    return None


def compute_fwhm_and_edge_sharpness(s_mm: np.ndarray, profile: np.ndarray) -> Dict[str, float]:
    profile = np.asarray(profile, dtype=float)
    s_mm = np.asarray(s_mm, dtype=float)
    if profile.size == 0 or profile.max() <= 0:
        return {
            'fwhm_mm': np.nan,
            'edge_left_25_75_mm': np.nan,
            'edge_right_25_75_mm': np.nan,
            'peak_mm': np.nan,
        }

    p = profile / profile.max()
    peak_idx = int(np.argmax(p))
    peak_x = float(s_mm[peak_idx])

    left_x = s_mm[: peak_idx + 1]
    left_p = p[: peak_idx + 1]
    right_x = s_mm[peak_idx:]
    right_p = p[peak_idx:]

    x50_left = first_crossing_position(left_x, left_p, 0.5, side='left')
    x50_right = first_crossing_position(right_x, right_p, 0.5, side='right')
    fwhm_mm = np.nan
    if x50_left is not None and x50_right is not None:
        fwhm_mm = float(x50_right - x50_left)

    x25_left = first_crossing_position(left_x, left_p, 0.25, side='left')
    x75_left = first_crossing_position(left_x, left_p, 0.75, side='left')
    edge_left = np.nan
    if x25_left is not None and x75_left is not None:
        edge_left = float(abs(x75_left - x25_left))

    x75_right = first_crossing_position(right_x, right_p, 0.75, side='right')
    x25_right = first_crossing_position(right_x, right_p, 0.25, side='right')
    edge_right = np.nan
    if x75_right is not None and x25_right is not None:
        edge_right = float(abs(x25_right - x75_right))

    return {
        'fwhm_mm': fwhm_mm,
        'edge_left_25_75_mm': edge_left,
        'edge_right_25_75_mm': edge_right,
        'peak_mm': peak_x,
    }


def centroid_of_component(weighted_volume: ArrayLike, mask: ArrayLike) -> np.ndarray:
    comp = np.asarray(weighted_volume, dtype=float) * mask.astype(float)
    if comp.sum() <= 0:
        idx = np.argwhere(mask)
        if idx.size == 0:
            raise ValueError('Empty component mask')
        return idx.mean(axis=0)
    return np.asarray(center_of_mass(comp))


def analyze_single_component(
    volume: ArrayLike,
    component_mask: ArrayLike,
    spacing_zyx: Sequence[float],
    analysis_axis_zyx: Sequence[float],
) -> Dict[str, float]:
    spacing_zyx = np.asarray(spacing_zyx, dtype=float)
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
    metrics['centroid_z_vox'] = float(center[0])
    metrics['centroid_y_vox'] = float(center[1])
    metrics['centroid_x_vox'] = float(center[2])
    return metrics


def save_profile_png(
    s_mm: np.ndarray,
    profile: np.ndarray,
    png_path: Path,
    title: str,
) -> None:
    p = np.asarray(profile, dtype=float)
    if p.max() > 0:
        p = p / p.max()
    fig = plt.figure(figsize=(7, 4.5))
    ax = fig.add_subplot(111)
    ax.plot(s_mm, p, linewidth=2)
    ax.axhline(0.5, linestyle='--', linewidth=1)
    ax.axhline(0.25, linestyle=':', linewidth=1)
    ax.axhline(0.75, linestyle=':', linewidth=1)
    ax.set_xlabel('Position along analysis axis (mm)')
    ax.set_ylabel('Normalized profile')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(png_path, dpi=180)
    plt.close(fig)


def save_3d_png(
    volume: ArrayLike,
    component_mask: ArrayLike,
    spacing_zyx: Sequence[float],
    analysis_axis_zyx: Sequence[float],
    png_path: Path,
    title: str,
    render_threshold_fraction: float = 0.10,
) -> None:
    spacing_zyx = np.asarray(spacing_zyx, dtype=float)
    vol = np.asarray(volume, dtype=float)
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

    axis = unit_vector(np.asarray(analysis_axis_zyx, dtype=float))
    axis_xyz = np.array([axis[2] * spacing_zyx[2], axis[1] * spacing_zyx[1], axis[0] * spacing_zyx[0]])
    axis_xyz = unit_vector(axis_xyz)
    line_half = 0.25 * np.linalg.norm(np.asarray(vol.shape) * spacing_zyx)
    line = np.vstack([center_xyz - line_half * axis_xyz, center_xyz + line_half * axis_xyz])

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=vals, s=8, alpha=0.45)
    ax.plot(line[:, 0], line[:, 1], line[:, 2], linewidth=2)
    ax.scatter([center_xyz[0]], [center_xyz[1]], [center_xyz[2]], s=50, marker='x')
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_zlabel('z (mm)')
    ax.set_title(title)
    fig.colorbar(scatter, ax=ax, shrink=0.7, pad=0.08, label='PPDF intensity')
    fig.tight_layout()
    fig.savefig(png_path, dpi=180)
    plt.close(fig)


def choose_components(volume: ArrayLike, support_threshold_fraction: float, min_component_voxels: int) -> Tuple[ArrayLike, int]:
    vol = normalize_volume(volume)
    mask = vol >= support_threshold_fraction
    labels, n = connected_components(mask)
    if n == 0:
        return labels, 0

    kept = np.zeros_like(labels)
    new_id = 0
    for cid in range(1, n + 1):
        comp = labels == cid
        if int(comp.sum()) >= min_component_voxels:
            new_id += 1
            kept[comp] = new_id
    return kept, new_id


def analyze_ppdf_row(
    volume: ArrayLike,
    spacing_zyx: Sequence[float],
    analysis_axis_zyx: Sequence[float],
    support_threshold_fraction: float = 0.10,
    min_component_voxels: int = 5,
) -> Dict[str, object]:
    vol = normalize_volume(volume)
    labels, ncomp = choose_components(vol, support_threshold_fraction, min_component_voxels)

    components: List[Dict[str, float]] = []
    profiles: List[Tuple[np.ndarray, np.ndarray]] = []
    for cid in range(1, ncomp + 1):
        mask = labels == cid
        m = analyze_single_component(vol, mask, spacing_zyx, analysis_axis_zyx)
        center = np.array([m['centroid_z_vox'], m['centroid_y_vox'], m['centroid_x_vox']])
        s_mm, profile = extract_line_profile(vol * mask.astype(float), center, analysis_axis_zyx, spacing_zyx)
        components.append(m)
        profiles.append((s_mm, profile))

    return {
        'normalized_volume': vol,
        'component_labels': labels,
        'components': components,
        'profiles': profiles,
    }


def save_population_histograms(metrics: List[Dict[str, float]], png_path: Path) -> None:
    if not metrics:
        return

    fwhm = np.array([m['fwhm_mm'] for m in metrics], dtype=float)
    edge = np.array([np.nanmean([m['edge_left_25_75_mm'], m['edge_right_25_75_mm']]) for m in metrics], dtype=float)
    active_vol = np.array([m['active_volume_mm3'] for m in metrics], dtype=float)

    fig = plt.figure(figsize=(12, 3.8))
    axes = [fig.add_subplot(1, 3, i + 1) for i in range(3)]
    axes[0].hist(fwhm[np.isfinite(fwhm)], bins=24)
    axes[0].set_title('FWHM distribution')
    axes[0].set_xlabel('mm')

    axes[1].hist(edge[np.isfinite(edge)], bins=24)
    axes[1].set_title('Edge 25-75 distribution')
    axes[1].set_xlabel('mm')

    axes[2].hist(active_vol[np.isfinite(active_vol)], bins=24)
    axes[2].set_title('Active volume distribution')
    axes[2].set_xlabel('mm^3')

    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(png_path, dpi=180)
    plt.close(fig)


def save_detector_metric_map(
    metrics: List[Dict[str, float]],
    detector_positions_xyz_mm: np.ndarray,
    metric_key: str,
    png_path: Path,
    title: str,
) -> None:
    vals = np.array([m.get(metric_key, np.nan) for m in metrics], dtype=float)
    valid = np.isfinite(vals)
    pts = np.asarray(detector_positions_xyz_mm, dtype=float)
    if pts.shape[0] != vals.shape[0]:
        raise ValueError('detector_positions_xyz_mm length must match number of metric entries.')
    pts = pts[valid]
    vals = vals[valid]
    if pts.size == 0:
        return

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=vals, s=28)
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_zlabel('z (mm)')
    ax.set_title(title)
    fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.08, label=metric_key)
    fig.tight_layout()
    fig.savefig(png_path, dpi=180)
    plt.close(fig)


def metrics_to_json_serializable(metrics: List[Dict[str, float]]) -> List[Dict[str, float]]:
    out = []
    for m in metrics:
        clean = {}
        for k, v in m.items():
            if isinstance(v, (np.floating, np.integer)):
                clean[k] = float(v)
            else:
                clean[k] = v
        out.append(clean)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description='PPDF beam-shape metrics for 3D system-matrix rows.')
    parser.add_argument('--input', required=True, help='Path to .npy file. Shape can be (rows, z, y, x) or (z, y, x).')
    parser.add_argument('--output_dir', required=True, help='Directory where PNGs/JSON will be written.')
    parser.add_argument('--voxel_size_mm', nargs=3, type=float, required=True, metavar=('Z', 'Y', 'X'))
    parser.add_argument('--analysis_axis_zyx', nargs=3, type=float, default=[0.0, 0.0, 1.0], metavar=('AZ', 'AY', 'AX'))
    parser.add_argument('--rows', nargs='*', type=int, default=None, help='Rows to analyze for a 4D system matrix. Default: all rows.')
    parser.add_argument('--support_threshold_fraction', type=float, default=0.10)
    parser.add_argument('--min_component_voxels', type=int, default=5)
    parser.add_argument('--detector_positions_xyz_mm', default=None, help='Optional .npy file of detector positions for metric-map PNGs.')
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    arr = np.load(input_path)
    if arr.ndim == 3:
        arr = arr[None, ...]
    if arr.ndim != 4:
        raise ValueError('Input array must have shape (rows, z, y, x) or (z, y, x).')

    spacing = voxel_spacing_zyx(args.voxel_size_mm)
    axis = unit_vector(args.analysis_axis_zyx)

    rows = args.rows if args.rows is not None and len(args.rows) > 0 else list(range(arr.shape[0]))
    all_metrics: List[Dict[str, float]] = []
    detector_metric_rows: List[Dict[str, float]] = []

    for row_idx in rows:
        result = analyze_ppdf_row(
            arr[row_idx],
            spacing_zyx=spacing,
            analysis_axis_zyx=axis,
            support_threshold_fraction=args.support_threshold_fraction,
            min_component_voxels=args.min_component_voxels,
        )
        labels = result['component_labels']
        comps = result['components']
        profiles = result['profiles']

        for local_id, (comp_metrics, (s_mm, profile)) in enumerate(zip(comps, profiles), start=1):
            comp_metrics = dict(comp_metrics)
            comp_metrics['row_idx'] = row_idx
            comp_metrics['component_idx'] = local_id
            all_metrics.append(comp_metrics)

            comp_mask = labels == local_id
            base = output_dir / f'row_{row_idx:04d}_comp_{local_id:02d}'
            save_3d_png(
                result['normalized_volume'],
                comp_mask,
                spacing,
                axis,
                base.with_name(base.name + '_3d.png'),
                title=f'Row {row_idx} | Component {local_id}',
            )
            save_profile_png(
                s_mm,
                profile,
                base.with_name(base.name + '_profile.png'),
                title=f'Row {row_idx} | Component {local_id} profile',
            )

        # One representative row-level record for optional detector-position maps.
        if comps:
            row_fwhm = np.nanmean([m['fwhm_mm'] for m in comps])
            row_edge = np.nanmean([
                np.nanmean([m['edge_left_25_75_mm'], m['edge_right_25_75_mm']])
                for m in comps
            ])
            row_active = np.nansum([m['active_volume_mm3'] for m in comps])
        else:
            row_fwhm = np.nan
            row_edge = np.nan
            row_active = np.nan
        detector_metric_rows.append({
            'row_idx': row_idx,
            'fwhm_mm': row_fwhm,
            'edge_25_75_mm': row_edge,
            'active_volume_mm3': row_active,
        })

    save_population_histograms(all_metrics, output_dir / 'population_histograms.png')

    if args.detector_positions_xyz_mm is not None:
        det_pos = np.load(args.detector_positions_xyz_mm)
        if det_pos.ndim != 2 or det_pos.shape[1] != 3:
            raise ValueError('detector_positions_xyz_mm must be a .npy file with shape (n_rows, 3).')
        selected_positions = det_pos[rows]
        save_detector_metric_map(
            detector_metric_rows,
            selected_positions,
            metric_key='fwhm_mm',
            png_path=output_dir / 'detector_fwhm_map_3d.png',
            title='Row-wise mean FWHM over detector geometry',
        )
        save_detector_metric_map(
            detector_metric_rows,
            selected_positions,
            metric_key='edge_25_75_mm',
            png_path=output_dir / 'detector_edge_map_3d.png',
            title='Row-wise mean edge sharpness over detector geometry',
        )
        save_detector_metric_map(
            detector_metric_rows,
            selected_positions,
            metric_key='active_volume_mm3',
            png_path=output_dir / 'detector_active_volume_map_3d.png',
            title='Row-wise active volume over detector geometry',
        )

    with open(output_dir / 'metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics_to_json_serializable(all_metrics), f, indent=2)

    with open(output_dir / 'row_summary.json', 'w', encoding='utf-8') as f:
        json.dump(metrics_to_json_serializable(detector_metric_rows), f, indent=2)


if __name__ == '__main__':
    main()
