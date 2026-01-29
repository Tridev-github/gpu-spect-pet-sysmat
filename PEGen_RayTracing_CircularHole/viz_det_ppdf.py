#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")  # terminal-safe
import matplotlib.pyplot as plt
import os

def read_f32(path):
    return np.fromfile(path, dtype=np.float32)

def load_image_params(path):
    # Your generator writes 12 float32s:
    # [NX,NY,NZ, vx,vy,vz, numRot, anglePerRot, shiftx,shifty,shiftz, FOV2COLL0]
    a = read_f32(path)
    if a.size < 6:
        raise RuntimeError(f"{path} too small to contain image params")
    nx, ny, nz = int(a[0]), int(a[1]), int(a[2]) if a.size >= 3 else 1
    vx, vy, vz = float(a[3]), float(a[4]), float(a[5])
    return nx, ny, nz, vx, vy, vz

def load_detector_centers(path):
    a = read_f32(path)
    if a.size < 1:
        raise RuntimeError(f"{path} empty")
    num_det = int(round(float(a[0])))
    expected = 1 + 12 * num_det
    if a.size != expected:
        raise RuntimeError(
            f"{path} size mismatch: got {a.size} float32s, expected {expected} (=1+12*num_det). "
            f"num_det(from file)={num_det}"
        )
    det = a[1:].reshape(num_det, 12)
    x = det[:, 0]
    y = det[:, 1]
    z = det[:, 2]
    return num_det, x, y, z, det

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sysmat", default="PE_SysMat_shift_0.000000_0.000000_0.000000.sysmat")
    ap.add_argument("--image_params", default="Params_Image.dat")
    ap.add_argument("--detector_params", default="Params_Detector.dat")
    ap.add_argument("--det", type=int, default=0, help="selected detector index [0..numDet-1]")
    ap.add_argument("--rows", type=int, default=5632)
    ap.add_argument("--cols", type=int, default=25600)
    ap.add_argument("--out", default="det0_ppdf_overlay.png")
    ap.add_argument("--log", action="store_true", help="log1p scale PPDF for visibility")
    ap.add_argument("--alpha", type=float, default=0.70, help="PPDF overlay alpha")
    args = ap.parse_args()

    # Sanity: sysmat file exists
    if not os.path.exists(args.sysmat):
        raise FileNotFoundError(args.sysmat)

    # Load image params (binary)
    nx, ny, nz, vx, vy, vz = load_image_params(args.image_params)
    if nz != 1:
        # Your sysmat cols=nx*ny*nz; visualization below assumes a 2D plane.
        # We can still visualize a slice later if needed.
        pass

    # Confirm sysmat cols match FOV voxels (your case: 160*160*1=25600)
    fov_vox = nx * ny * nz
    if fov_vox != args.cols:
        raise RuntimeError(f"FOV voxels (nx*ny*nz={fov_vox}) != sysmat cols ({args.cols}). "
                           f"Fix either Params_Image.dat or pass matching --cols.")

    # Load detectors (binary)
    numDet, dx, dy, dz, det_full = load_detector_centers(args.detector_params)
    if numDet != args.rows:
        raise RuntimeError(f"Detector count mismatch: Params_Detector.dat says {numDet}, but you passed --rows {args.rows}")

    if not (0 <= args.det < numDet):
        raise ValueError(f"--det must be in [0..{numDet-1}]")

    # Memmap sysmat and load selected PPDF row
    A = np.memmap(args.sysmat, dtype=np.float32, mode="r", shape=(args.rows, args.cols))
    p = np.array(A[args.det, :], dtype=np.float32).reshape((nx, ny))  # your generator uses 160x160x1
    p_disp = np.log1p(p) if args.log else p

    # Coordinate mapping:
    # Voxel indices i=0..nx-1 map to x in mm: x = (i - (nx-1)/2)*vx
    # Voxel indices j=0..ny-1 map to z in mm: z = (j - (ny-1)/2)*vz
    x_min = -(nx - 1) / 2 * vx
    x_max =  (nx - 1) / 2 * vx
    z_min = -(ny - 1) / 2 * vz
    z_max =  (ny - 1) / 2 * vz

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1)

    # Plot PPDF in physical mm coordinates
    # Note: p is [x_index, z_index] effectively; transpose so x is horizontal, z is vertical.
    im = ax.imshow(
        p_disp.T,
        origin="lower",
        extent=[x_min, x_max, z_min, z_max],
        aspect="equal",
        alpha=args.alpha
    )

    # Overlay detector geometry projected into x–z plane
    ax.scatter(dx, dz, s=8, marker="o", alpha=0.35)

    # Highlight selected detector
    ax.scatter([dx[args.det]], [dz[args.det]], s=120, marker="o")

    ax.set_title(f"Detector geometry (x–z) + PPDF overlay | det={args.det} | nx={nx},ny={ny},vx={vx},vz={vz}" +
                 (" | log1p" if args.log else ""))
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("z (mm)")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("PPDF (log1p)" if args.log else "PPDF")

    fig.tight_layout()
    fig.savefig(args.out, dpi=220)
    print(f"WROTE {args.out}")

if __name__ == "__main__":
    main()
