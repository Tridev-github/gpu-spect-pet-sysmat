#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt

def read_f32(path):
    return np.fromfile(path, dtype=np.float32)

def pick_6_indices(n):
    # 0th, last, and 4 evenly spaced between
    if n <= 1:
        return [0]
    idx = np.linspace(0, n-1, 6)
    idx = np.round(idx).astype(int)
    # ensure unique and sorted
    idx = np.unique(idx)
    return idx.tolist()

def save_6_slices(img3d, plane, out_png):
    """
    img3d: (nx, ny, nz)
    plane: "XY", "XZ", "YZ"
    """
    nx, ny, nz = img3d.shape

    if plane == "XY":
        ks = pick_6_indices(nz)
        ncols = len(ks)
        fig, axes = plt.subplots(1, ncols, figsize=(3*ncols, 3))
        if ncols == 1:
            axes = [axes]
        for ax, k in zip(axes, ks):
            ax.imshow(img3d[:, :, k].T, origin="lower")
            ax.set_title(f"z={k}")
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close(fig)
        return

    if plane == "XZ":
        js = pick_6_indices(ny)
        ncols = len(js)
        fig, axes = plt.subplots(1, ncols, figsize=(3*ncols, 3))
        if ncols == 1:
            axes = [axes]
        for ax, j in zip(axes, js):
            ax.imshow(img3d[:, j, :].T, origin="lower")
            ax.set_title(f"y={j}")
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close(fig)
        return

    if plane == "YZ":
        is_ = pick_6_indices(nx)
        ncols = len(is_)
        fig, axes = plt.subplots(1, ncols, figsize=(3*ncols, 3))
        if ncols == 1:
            axes = [axes]
        for ax, i in zip(axes, is_):
            ax.imshow(img3d[i, :, :].T, origin="lower")
            ax.set_title(f"x={i}")
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close(fig)
        return

    raise ValueError("plane must be XY, XZ, or YZ")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sysmat", required=True, help=".sysmat float32 file from PEGen")
    ap.add_argument("--param_detector", default="Param_Detector.dat")
    ap.add_argument("--param_image", default="Param_Image.dat")
    ap.add_argument("--det_id", type=int, default=0)
    ap.add_argument("--out_prefix", default="ppdf")
    args = ap.parse_args()

    img = read_f32(args.param_image)
    nx, ny, nz = int(img[0]), int(img[1]), int(img[2])

    det = read_f32(args.param_detector)
    num_det = int(det[0])

    num_vox = nx * ny * nz

    sm = read_f32(args.sysmat)

    # Try to infer layout:
    # Most common: stored as float32 length = num_det * num_vox (single rotation)
    expected = num_det * num_vox
    if sm.size < expected:
        raise RuntimeError(
            f"Sysmat too small: got {sm.size} floats, expected at least {expected} "
            f"(num_det={num_det}, vox={num_vox})."
        )

    # take first rotation worth if extra exists
    sm = sm[:expected]
    sm2 = sm.reshape((num_det, num_vox))

    if args.det_id < 0 or args.det_id >= num_det:
        raise ValueError(f"det_id must be in [0, {num_det-1}]")

    ppdf = sm2[args.det_id, :]  # PPDF: per-voxel probs for this detector bin

    # reshape into volume: assume x fastest, then y, then z (C-order)
    vol_zyx = ppdf.reshape((nz, ny, nx))
    vol = np.transpose(vol_zyx, (2, 1, 0))  # -> (nx, ny, nz)

    nnz = int(np.count_nonzero(vol))
    print(f"PPDF stats: min={vol.min():.3e} max={vol.max():.3e} nnz={nnz} shape={vol.shape}")

    save_6_slices(vol, "XY", f"{args.out_prefix}_XY_6slices.png")
    print(f"Wrote {args.out_prefix}_XY_6slices.png")

    save_6_slices(vol, "XZ", f"{args.out_prefix}_XZ_6slices.png")
    print(f"Wrote {args.out_prefix}_XZ_6slices.png")

    save_6_slices(vol, "YZ", f"{args.out_prefix}_YZ_6slices.png")
    print(f"Wrote {args.out_prefix}_YZ_6slices.png")

if __name__ == "__main__":
    main()
