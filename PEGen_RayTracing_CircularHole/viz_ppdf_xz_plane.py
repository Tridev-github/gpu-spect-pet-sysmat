import argparse, os
import numpy as np
import matplotlib.pyplot as plt

def load_params_image(path):
    img = np.fromfile(path, np.float32)
    nx, ny, nz = map(int, img[:3])
    return nx, ny, nz

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sysmat", required=True)
    ap.add_argument("--param_detector", required=True)
    ap.add_argument("--param_image", required=True)
    ap.add_argument("--det_id", type=int, required=True)
    ap.add_argument("--out", default="ppdf_xz.png")
    ap.add_argument("--log", action="store_true")
    args = ap.parse_args()

    det = np.fromfile(args.param_detector, np.float32)
    num_det = int(det[0])

    nx, ny, nz = load_params_image(args.param_image)
    vox = nx * ny * nz

    if args.det_id < 0 or args.det_id >= num_det:
        raise SystemExit(f"det_id out of range [0,{num_det-1}]")

    floats = os.path.getsize(args.sysmat) // 4
    expected = num_det * vox
    if floats < expected:
        raise RuntimeError(f"Sysmat too small: got {floats} floats, expected {expected}")

    mm = np.memmap(args.sysmat, dtype=np.float32, mode="r", shape=(num_det, vox))
    row = np.array(mm[args.det_id], dtype=np.float32)

    # For X-Z plane convention: (nx, ny=1, nz) -> reshape to (nz, nx) for imshow (rows=z, cols=x)
    vol = row.reshape((nz, ny, nx))  # z, y, x
    plane = vol[:, 0, :]             # z, x

    nnz = np.count_nonzero(plane)
    print(f"PPDF plane stats: min={plane.min():.3e} max={plane.max():.3e} nnz={nnz} shape={plane.shape}")

    img = np.log10(np.maximum(plane, 1e-30)) if args.log else plane

    plt.figure(figsize=(6,6))
    plt.imshow(img, origin="lower", aspect="equal")
    plt.colorbar(label="log10(PPDF)" if args.log else "PPDF")
    plt.title(f"det_id={args.det_id}  X-Z plane (ny=1)")
    plt.xlabel("X index")
    plt.ylabel("Z index")
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print("Wrote", args.out)

if __name__ == "__main__":
    main()
