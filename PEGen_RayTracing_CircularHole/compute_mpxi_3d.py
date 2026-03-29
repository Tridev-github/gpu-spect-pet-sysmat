import os
import argparse
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# DEFAULT CONFIG
# ============================================================
MASK_H5 = "beams_masks_3d.hdf5"

NUM_ROT = 60
NUM_DET = 5632
VOX_SHAPE = (16, 16, 16)
NUM_VOX = np.prod(VOX_SHAPE)

DEFAULT_OUTDIR = "/eng/home/tridevme/GPU-Based-System-Matrix-Calculation-for-SPECT-PET/PEGen_RayTracing_CircularHole/beam_masks/mpxi_3d"


# ============================================================
# HELPERS
# ============================================================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def global_index(rot_idx, det_idx):
    return rot_idx * NUM_DET + det_idx


def load_rotation_masks(mask_ds, rot_idx):
    """
    Loads all detector masks for one rotation.
    Returns shape: (NUM_DET, NUM_VOX)
    """
    start = rot_idx * NUM_DET
    end = start + NUM_DET
    arr = np.array(mask_ds[start:end, :], dtype=np.uint8)
    return arr


def compute_mpxi_for_rotation(mask_ds, rot_idx):
    """
    MPXI for one rotation:
    For each voxel, count how many detector-beam masks include that voxel.
    """
    rot_masks = load_rotation_masks(mask_ds, rot_idx)   # shape: (NUM_DET, NUM_VOX)
    mpxi_flat = np.sum(rot_masks, axis=0, dtype=np.uint32)  # shape: (NUM_VOX,)
    return mpxi_flat.reshape(VOX_SHAPE)


def summarize_mpxi(mpxi_3d):
    vals = mpxi_3d.ravel()
    nonzero = vals[vals > 0]

    summary = {
        "shape": tuple(mpxi_3d.shape),
        "num_voxels_total": int(vals.size),
        "num_voxels_sampled": int(np.sum(vals > 0)),
        "num_voxels_unsampled": int(np.sum(vals == 0)),
        "num_voxels_nonmultiplexed": int(np.sum(vals == 1)),
        "num_voxels_multiplexed": int(np.sum(vals > 1)),
        "max_mpxi": int(np.max(vals)),
        "mean_mpxi_all_voxels": float(np.mean(vals)),
        "mean_mpxi_sampled_voxels": float(np.mean(nonzero)) if nonzero.size > 0 else 0.0,
        "median_mpxi_sampled_voxels": float(np.median(nonzero)) if nonzero.size > 0 else 0.0,
    }

    unique_vals, counts = np.unique(vals, return_counts=True)
    summary["histogram"] = list(zip(unique_vals.tolist(), counts.tolist()))
    return summary


def save_summary_txt(summary, save_path, rot_idx):
    with open(save_path, "w") as f:
        f.write(f"3D MPXI Summary | Rotation {rot_idx}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Volume shape: {summary['shape']}\n")
        f.write(f"Total voxels: {summary['num_voxels_total']}\n")
        f.write(f"Sampled voxels (MPXI > 0): {summary['num_voxels_sampled']}\n")
        f.write(f"Unsampled voxels (MPXI = 0): {summary['num_voxels_unsampled']}\n")
        f.write(f"Non-multiplexed voxels (MPXI = 1): {summary['num_voxels_nonmultiplexed']}\n")
        f.write(f"Multiplexed voxels (MPXI > 1): {summary['num_voxels_multiplexed']}\n")
        f.write(f"Max MPXI: {summary['max_mpxi']}\n")
        f.write(f"Mean MPXI (all voxels): {summary['mean_mpxi_all_voxels']:.4f}\n")
        f.write(f"Mean MPXI (sampled voxels): {summary['mean_mpxi_sampled_voxels']:.4f}\n")
        f.write(f"Median MPXI (sampled voxels): {summary['median_mpxi_sampled_voxels']:.4f}\n\n")

        f.write("Histogram (MPXI value -> voxel count)\n")
        f.write("-" * 60 + "\n")
        for val, cnt in summary["histogram"]:
            f.write(f"{val:>6} -> {cnt}\n")


def save_slice_panel(mpxi_3d, save_path, rot_idx):
    cx, cy, cz = [s // 2 for s in mpxi_3d.shape]

    xy = mpxi_3d[:, :, cz]
    xz = mpxi_3d[:, cy, :]
    yz = mpxi_3d[cx, :, :]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    items = [
        (xy, f"XY slice (z={cz})"),
        (xz, f"XZ slice (y={cy})"),
        (yz, f"YZ slice (x={cx})"),
    ]

    for ax, (img, title) in zip(axes, items):
        im = ax.imshow(img.T, origin="lower", aspect="auto")
        ax.set_title(title)
        ax.set_xlabel("Voxel index")
        ax.set_ylabel("Voxel index")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"3D MPXI Central Slices | Rotation {rot_idx}", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close(fig)


def save_mip_panel(mpxi_3d, save_path, rot_idx):
    xy_mip = np.max(mpxi_3d, axis=2)
    xz_mip = np.max(mpxi_3d, axis=1)
    yz_mip = np.max(mpxi_3d, axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    items = [
        (xy_mip, "Max Projection XY"),
        (xz_mip, "Max Projection XZ"),
        (yz_mip, "Max Projection YZ"),
    ]

    for ax, (img, title) in zip(axes, items):
        im = ax.imshow(img.T, origin="lower", aspect="auto")
        ax.set_title(title)
        ax.set_xlabel("Voxel index")
        ax.set_ylabel("Voxel index")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"3D MPXI Maximum Intensity Projections | Rotation {rot_idx}", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close(fig)


def save_histogram_plot(mpxi_3d, save_path, rot_idx):
    vals = mpxi_3d.ravel()
    unique_vals, counts = np.unique(vals, return_counts=True)

    plt.figure(figsize=(10, 5))
    plt.bar(unique_vals, counts)
    plt.xlabel("MPXI value")
    plt.ylabel("Number of voxels")
    plt.title(f"MPXI Histogram | Rotation {rot_idx}")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()


def save_nonzero_histogram_plot(mpxi_3d, save_path, rot_idx):
    vals = mpxi_3d.ravel()
    vals = vals[vals > 0]

    if vals.size == 0:
        print(f"Skipping nonzero histogram: no sampled voxels.")
        return

    unique_vals, counts = np.unique(vals, return_counts=True)

    plt.figure(figsize=(10, 5))
    plt.bar(unique_vals, counts)
    plt.xlabel("MPXI value")
    plt.ylabel("Number of sampled voxels")
    plt.title(f"MPXI Histogram (Sampled Voxels Only) | Rotation {rot_idx}")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()


def save_h5_volume(mpxi_3d, save_path, rot_idx):
    with h5py.File(save_path, "w") as f:
        f.create_dataset("mpxi_3d", data=mpxi_3d, dtype=np.uint32)
        f.attrs["rotation_index"] = int(rot_idx)
        f.attrs["definition"] = "MPXI(voxel) = number of detector-beam masks covering that voxel for the given rotation"


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Compute 3D MPXI from beam masks.")
    parser.add_argument("--mask-h5", type=str, default=MASK_H5, help="Path to beams_masks_3d.hdf5")
    parser.add_argument("--rot", type=int, default=0, help="Rotation index")
    parser.add_argument("--outdir", type=str, default=DEFAULT_OUTDIR, help="Output directory")
    parser.add_argument("--all-rots", action="store_true", help="Compute MPXI for all rotations")
    args = parser.parse_args()

    if not os.path.exists(args.mask_h5):
        raise FileNotFoundError(f"Mask HDF5 file not found: {args.mask_h5}")

    ensure_dir(args.outdir)

    with h5py.File(args.mask_h5, "r") as fmask:
        mask_ds = fmask["beam_mask"]

        rotations = range(NUM_ROT) if args.all_rots else [args.rot]

        for rot_idx in rotations:
            if not (0 <= rot_idx < NUM_ROT):
                raise ValueError(f"Rotation index must be in [0, {NUM_ROT - 1}]")

            print(f"Computing 3D MPXI for rotation {rot_idx}...")

            mpxi_3d = compute_mpxi_for_rotation(mask_ds, rot_idx)
            summary = summarize_mpxi(mpxi_3d)

            base = f"mpxi_3d_rot_{rot_idx:03d}"

            npy_path = os.path.join(args.outdir, f"{base}.npy")
            h5_path = os.path.join(args.outdir, f"{base}.hdf5")
            txt_path = os.path.join(args.outdir, f"{base}_summary.txt")
            slices_path = os.path.join(args.outdir, f"{base}_slices.png")
            mip_path = os.path.join(args.outdir, f"{base}_mips.png")
            hist_path = os.path.join(args.outdir, f"{base}_hist_all.png")
            hist_nonzero_path = os.path.join(args.outdir, f"{base}_hist_nonzero.png")

            np.save(npy_path, mpxi_3d)
            save_h5_volume(mpxi_3d, h5_path, rot_idx)
            save_summary_txt(summary, txt_path, rot_idx)
            save_slice_panel(mpxi_3d, slices_path, rot_idx)
            save_mip_panel(mpxi_3d, mip_path, rot_idx)
            save_histogram_plot(mpxi_3d, hist_path, rot_idx)
            save_nonzero_histogram_plot(mpxi_3d, hist_nonzero_path, rot_idx)

            print(f"Saved:")
            print(f"  {npy_path}")
            print(f"  {h5_path}")
            print(f"  {txt_path}")
            print(f"  {slices_path}")
            print(f"  {mip_path}")
            print(f"  {hist_path}")
            print(f"  {hist_nonzero_path}")

    print("Done.")


if __name__ == "__main__":
    main()
