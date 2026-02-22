#!/usr/bin/env python3
import argparse, os, sys
import numpy as np

def parse_detectors(num_det, det_list, det_every, det_count, det_start, det_end):
    if det_list:
        dets = [int(x) for x in det_list.split(",") if x.strip() != ""]
        dets = [d for d in dets if 0 <= d < num_det]
        if not dets:
            raise SystemExit("[ERROR] --det_list produced no valid detector indices.")
        return dets
    if det_every is not None:
        step = int(det_every)
        if step <= 0:
            raise SystemExit("[ERROR] --det_every must be > 0")
        return list(range(0, num_det, step))[:int(det_count)]
    s = max(0, int(det_start))
    e = num_det if det_end is None else min(num_det, int(det_end))
    if s >= e:
        raise SystemExit("[ERROR] Empty detector range")
    return list(range(s, e))

def main():
    ap = argparse.ArgumentParser(description="Plot 3D boundary (isosurface) of per-detector PPDF using Matplotlib3D.")
    ap.add_argument("--sysmat", required=True, help="Path to .sysmat (dense float32)")
    ap.add_argument("--nx", type=int, required=True)
    ap.add_argument("--ny", type=int, required=True)
    ap.add_argument("--nz", type=int, required=True)
    ap.add_argument("--num_det", type=int, required=True)
    ap.add_argument("--order", choices=["det_major", "vox_major"], default="det_major",
                    help="Memory layout: det_major means [det][vox].")

    ap.add_argument("--out_dir", required=True, help="Folder to save PNGs")
    ap.add_argument("--iso_frac", type=float, default=0.10,
                    help="Isosurface threshold as fraction of max (e.g., 0.1 = 10%% of max).")
    ap.add_argument("--downsample", type=int, default=1,
                    help="Downsample factor for volume before meshing (2 or 4 recommended for speed).")
    ap.add_argument("--max_faces", type=int, default=250000,
                    help="If mesh is too dense, it will be decimated by face subsampling to this count (rough).")

    ap.add_argument("--det_list", default=None)
    ap.add_argument("--det_every", type=int, default=None)
    ap.add_argument("--det_count", type=int, default=16)
    ap.add_argument("--det_start", type=int, default=0)
    ap.add_argument("--det_end", type=int, default=None)

    args = ap.parse_args()

    # Lazy imports that may not exist until installed
    try:
        from skimage.measure import marching_cubes
    except Exception as e:
        print("[ERROR] scikit-image not available. Install with: pip install --user scikit-image")
        raise

    import matplotlib
    matplotlib.use("Agg")  # headless save
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    nx, ny, nz, num_det = args.nx, args.ny, args.nz, args.num_det
    full_vox = nx * ny * nz

    if not os.path.exists(args.sysmat):
        raise SystemExit(f"[ERROR] sysmat not found: {args.sysmat}")

    expected = num_det * full_vox * 4
    actual = os.path.getsize(args.sysmat)
    if actual != expected:
        print("[ERROR] size mismatch")
        print(" expected:", expected, "bytes (num_det*nx*ny*nz*4)")
        print(" actual  :", actual, "bytes")
        sys.exit(2)

    os.makedirs(args.out_dir, exist_ok=True)

    mm = np.memmap(args.sysmat, dtype=np.float32, mode="r")
    if args.order == "det_major":
        mat = mm.reshape((num_det, full_vox))
        get_row = lambda d: mat[d, :]
    else:
        mat = mm.reshape((full_vox, num_det))
        get_row = lambda d: mat[:, d]

    dets = parse_detectors(num_det, args.det_list, args.det_every, args.det_count, args.det_start, args.det_end)

    ds = max(1, int(args.downsample))
    iso_frac = float(args.iso_frac)
    if not (0.0 < iso_frac < 1.0):
        raise SystemExit("[ERROR] --iso_frac must be between 0 and 1.")

    for j, d in enumerate(dets):
        row = get_row(d)
        vol = np.asarray(row, dtype=np.float32).reshape((nz, ny, nx))  # (Z,Y,X)

        # Downsample for speed (important: this reduces mesh complexity massively)
        if ds > 1:
            vol_ds = vol[::ds, ::ds, ::ds]
        else:
            vol_ds = vol

        vmax = float(vol_ds.max())
        if vmax <= 0:
            print(f"[WARN] det {d}: max<=0, skipping")
            continue

        level = vmax * iso_frac

        # marching cubes expects volume as (Z,Y,X). spacing accounts for downsample + voxel size(=1)
        verts, faces, normals, values = marching_cubes(vol_ds, level=level, spacing=(ds, ds, ds))

        # crude decimation if too many faces (matplotlib will choke)
        if faces.shape[0] > args.max_faces:
            step = int(np.ceil(faces.shape[0] / args.max_faces))
            faces = faces[::step, :]

        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection="3d")

        mesh = Poly3DCollection(verts[faces], alpha=0.35)
        ax.add_collection3d(mesh)

        # Set bounds
        ax.set_xlim(0, nx)
        ax.set_ylim(0, ny)
        ax.set_zlim(0, nz)

        ax.set_xlabel("X (vox)")
        ax.set_ylabel("Y (vox)")
        ax.set_zlabel("Z (vox)")
        ax.set_title(f"Detector {d} PPDF isosurface @ {iso_frac*100:.1f}% max (ds={ds})")

        # Equal-ish aspect (matplotlib 3D is annoying; this helps)
        try:
            ax.set_box_aspect((nx, ny, nz))
        except Exception:
            pass

        out_png = os.path.join(args.out_dir, f"det_{d:04d}_iso{int(iso_frac*100):02d}_ds{ds}.png")
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close(fig)

        if (j % 5) == 0:
            print(f"[OK] {j+1}/{len(dets)} saved {out_png}")

    print("[DONE]")

if __name__ == "__main__":
    main()
