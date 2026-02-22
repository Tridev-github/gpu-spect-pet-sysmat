#!/usr/bin/env python3
import argparse, os, sys, numpy as np

def safe_makedirs(p): os.makedirs(p, exist_ok=True)

def save_mips(vol_zyx, out_prefix):
    import matplotlib.pyplot as plt
    mip_xy = vol_zyx.max(axis=0)  # (Y,X)
    mip_xz = vol_zyx.max(axis=1)  # (Z,X)
    mip_yz = vol_zyx.max(axis=2)  # (Z,Y)

    def norm01(a):
        a = a.astype(np.float32, copy=False)
        mn, mx = float(a.min()), float(a.max())
        if mx <= mn: return np.zeros_like(a, dtype=np.float32)
        return (a - mn) / (mx - mn)

    for name, img in [("mip_xy", mip_xy), ("mip_xz", mip_xz), ("mip_yz", mip_yz)]:
        plt.figure()
        plt.imshow(norm01(img), origin="lower")
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.savefig(f"{out_prefix}_{name}.png", dpi=150, bbox_inches="tight", pad_inches=0)
        plt.close()

def parse_detectors(num_det, det_list, det_every, det_count, det_start, det_end):
    if det_list:
        dets = [int(x) for x in det_list.split(",") if x.strip()!=""]
        return [d for d in dets if 0 <= d < num_det]
    if det_every is not None:
        step = int(det_every)
        if step <= 0: raise SystemExit("--det_every must be > 0")
        return list(range(0, num_det, step))[:int(det_count)]
    s = max(0, int(det_start))
    e = num_det if det_end is None else min(num_det, int(det_end))
    if s >= e: raise SystemExit("Empty detector range")
    return list(range(s, e))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sysmat", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--nx", type=int, required=True)
    ap.add_argument("--ny", type=int, required=True)
    ap.add_argument("--nz", type=int, required=True)
    ap.add_argument("--num_det", type=int, required=True)
    ap.add_argument("--order", choices=["det_major","vox_major"], default="det_major")

    ap.add_argument("--det_list", default=None)
    ap.add_argument("--det_every", type=int, default=None)
    ap.add_argument("--det_count", type=int, default=64)
    ap.add_argument("--det_start", type=int, default=0)
    ap.add_argument("--det_end", type=int, default=None)

    ap.add_argument("--save_npy", action="store_true")
    ap.add_argument("--save_mip_png", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    nx, ny, nz, num_det = args.nx, args.ny, args.nz, args.num_det
    full_vox = nx*ny*nz

    actual = os.path.getsize(args.sysmat)
    expected = num_det*full_vox*4
    if actual != expected:
        print("[ERROR] size mismatch")
        print(" expected:", expected, "bytes")
        print(" actual  :", actual, "bytes")
        sys.exit(2)

    if not (args.save_npy or args.save_mip_png):
        print("[ERROR] choose --save_npy and/or --save_mip_png")
        sys.exit(2)

    safe_makedirs(args.out_dir)

    mm = np.memmap(args.sysmat, dtype=np.float32, mode="r")
    if args.order == "det_major":
        mat = mm.reshape((num_det, full_vox))
        get_row = lambda d: mat[d, :]
    else:
        mat = mm.reshape((full_vox, num_det))
        get_row = lambda d: mat[:, d]

    detectors = parse_detectors(num_det, args.det_list, args.det_every, args.det_count, args.det_start, args.det_end)

    for j, d in enumerate(detectors):
        out_base = os.path.join(args.out_dir, f"det_{d:04d}")
        row = get_row(d)
        vol_zyx = np.asarray(row, dtype=np.float32).reshape((nz, ny, nx))

        if args.save_npy:
            npy = out_base + ".npy"
            if args.overwrite or (not os.path.exists(npy)):
                np.save(npy, vol_zyx, allow_pickle=False)

        if args.save_mip_png:
            test = out_base + "_mip_xy.png"
            if args.overwrite or (not os.path.exists(test)):
                save_mips(vol_zyx, out_base)

        if (j % 25) == 0:
            print(f"[OK] {j+1}/{len(detectors)} exported (det={d})")

    print("[DONE]")

if __name__ == "__main__":
    main()
