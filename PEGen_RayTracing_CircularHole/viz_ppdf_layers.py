#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def read_f32(path): return np.fromfile(path, dtype=np.float32)

def load_image_params(path):
    a = read_f32(path)
    nx, ny, nz = int(a[0]), int(a[1]), int(a[2])
    vx, vy, vz = float(a[3]), float(a[4]), float(a[5])
    return nx, ny, nz, vx, vy, vz

def load_det(path):
    a = read_f32(path)
    n = int(round(float(a[0])))
    det = a[1:].reshape(n, 12)
    return n, det

def sysmat_row(sysmat_path, rows, cols, det_idx):
    A = np.memmap(sysmat_path, dtype=np.float32, mode="r", shape=(rows, cols))
    return np.array(A[det_idx, :], dtype=np.float32)

def find_nearest_det_in_layer(det_xyz, idxs, target_xz):
    x = det_xyz[idxs, 0]
    z = det_xyz[idxs, 2]
    tx, tz = target_xz
    j = np.argmin((x - tx)**2 + (z - tz)**2)
    return int(idxs[j])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sysmat", default="PE_SysMat_shift_0.000000_0.000000_0.000000.sysmat")
    ap.add_argument("--img", default="Params_Image.dat")
    ap.add_argument("--detpar", default="Params_Detector.dat")
    ap.add_argument("--out", default="mc_ppdf_layers.png")
    ap.add_argument("--det", type=int, default=0, help="detector index to anchor (in layer 1..4)")
    ap.add_argument("--rows", type=int, default=5632)
    ap.add_argument("--cols", type=int, default=25600)
    ap.add_argument("--log", action="store_true", help="use log10(PPDF)")
    ap.add_argument("--vmin", type=float, default=None)
    ap.add_argument("--vmax", type=float, default=None)
    ap.add_argument("--title", default="MC\nPPDF")
    args = ap.parse_args()

    nx, ny, nz, vx, vy, vz = load_image_params(args.img)
    if nx*ny*nz != args.cols:
        raise RuntimeError(f"FOV voxels nx*ny*nz={nx*ny*nz} != cols={args.cols}")

    nDet, det = load_det(args.detpar)
    if nDet != args.rows:
        raise RuntimeError(f"Params_Detector says {nDet} detectors, expected rows={args.rows}")
    if not (0 <= args.det < nDet):
        raise ValueError("bad --det")

    # Your generator order:
    # L1: 32*16=512
    # L2: 512
    # L3: 512
    # L4: 64*64=4096
    L123 = 32*16
    L4 = 64*64
    assert L123*3 + L4 == nDet, "detector count doesn't match expected layering"

    layer_ranges = [
        (0, L123),                 # Layer 1
        (L123, 2*L123),             # Layer 2
        (2*L123, 3*L123),           # Layer 3
        (3*L123, 3*L123 + L4)       # Layer 4
    ]

    det_xyz = det[:, [0,1,2]]  # x,y,z

    # Anchor physical x,z from selected detector
    anchor_x = float(det_xyz[args.det, 0])
    anchor_z = float(det_xyz[args.det, 2])
    target_xz = (anchor_x, anchor_z)

    # Choose one detector per layer:
    chosen = []
    for li, (a,b) in enumerate(layer_ranges):
        idxs = np.arange(a,b)
        if li < 3:
            # For layers 1-3, if anchor is within 32x16 set, use same ix/iz indexing when possible.
            # Otherwise, nearest in xz.
            chosen.append(find_nearest_det_in_layer(det_xyz, idxs, target_xz))
        else:
            # Layer 4: nearest in xz
            chosen.append(find_nearest_det_in_layer(det_xyz, idxs, target_xz))

    # Pull PPDFs, reshape to (nx,ny)
    pimgs = []
    for d in chosen:
        row = sysmat_row(args.sysmat, args.rows, args.cols, d).reshape((nx, ny))
        if args.log:
            eps = 1e-12
            img = np.log10(np.maximum(row, eps))
        else:
            img = row
        pimgs.append(img)

    # Color scale: match across panels unless user overrides
    allv = np.concatenate([im.ravel() for im in pimgs])
    vmin = np.min(allv) if args.vmin is None else args.vmin
    vmax = np.max(allv) if args.vmax is None else args.vmax

    # Plot layout: 4 rows, each with its own colorbar on the right
    fig = plt.figure(figsize=(5.2, 11.5))
    gs = fig.add_gridspec(4, 1, hspace=0.35)

    for i in range(4):
        ax = fig.add_subplot(gs[i,0])
        im = ax.imshow(pimgs[i].T, origin="lower", vmin=vmin, vmax=vmax)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_ylabel(f"Layer {i+1}", rotation=0, labelpad=35, va="center")

        # per-panel colorbar
        div = make_axes_locatable(ax)
        cax = div.append_axes("right", size="4.5%", pad=0.08)
        cb = fig.colorbar(im, cax=cax)
        cb.ax.tick_params(labelsize=8)
        if args.log:
            cb.set_label("log10", fontsize=8)

    fig.suptitle(args.title, y=0.99, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(args.out, dpi=250)
    print("WROTE", args.out)
    print("Chosen detectors per layer:", chosen)

if __name__ == "__main__":
    main()
