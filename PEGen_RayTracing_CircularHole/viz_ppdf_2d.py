#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt

def read_f32(path):
    return np.fromfile(path, dtype=np.float32)

def sysmat_stats(sys):
    mn = float(np.min(sys))
    mx = float(np.max(sys))
    nnz = int(np.count_nonzero(sys))
    neg = int(np.sum(sys < 0))
    return mn, mx, nnz, neg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sysmat", required=True)
    ap.add_argument("--param_detector", default="Param_Detector.dat")
    ap.add_argument("--param_image", default="Param_Image.dat")
    ap.add_argument("--det_id", type=int, default=0)
    ap.add_argument("--out", default="ppdf_det0.png")
    ap.add_argument("--log", action="store_true", help="log10(1e-30 + val) display")
    args = ap.parse_args()

    img = read_f32(args.param_image)
    det = read_f32(args.param_detector)

    nx, ny, nz = int(img[0]), int(img[1]), int(img[2])
    vox = nx * ny * nz
    num_det = int(det[0])

    sys = read_f32(args.sysmat)

    expected = num_det * vox  # assuming numRotation=1 in this run
    if sys.size < expected:
        raise RuntimeError(f"Sysmat too small: got {sys.size} floats, expected >= {expected} (num_det={num_det}, vox={vox}).")

    mn, mx, nnz, neg = sysmat_stats(sys[:expected])
    print(f"Sysmat stats: min={mn:.3e} max={mx:.3e} nnz={nnz} neg={neg} total={expected}")

    if args.det_id < 0 or args.det_id >= num_det:
        raise ValueError(f"det_id out of range 0..{num_det-1}")

    # Reshape: [det, vox]
    M = sys[:expected].reshape(num_det, vox)
    ppdf = M[args.det_id].reshape(nz, ny, nx)  # z,y,x
    p2d = ppdf[0]  # since nz=1 for 2D plane

    disp = p2d
    if args.log:
        disp = np.log10(np.maximum(disp, 1e-30))

    plt.figure(figsize=(7.5, 6))
    plt.imshow(disp, origin="lower")
    plt.title(f"PPDF (det {args.det_id})  nx={nx} ny={ny} nz={nz}" + (" [log10]" if args.log else ""))
    plt.xlabel("X index")
    plt.ylabel("Y index")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()
