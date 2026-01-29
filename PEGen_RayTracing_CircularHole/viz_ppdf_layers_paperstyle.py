#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable

def f32(path): return np.fromfile(path, dtype=np.float32)

def load_img_params(path):
    a = f32(path)
    nx, ny, nz = int(a[0]), int(a[1]), int(a[2])
    vx, vy, vz = float(a[3]), float(a[4]), float(a[5])
    return nx, ny, nz, vx, vy, vz

def load_det(path):
    a = f32(path)
    n = int(round(float(a[0])))
    det = a[1:].reshape(n, 12)
    return n, det

def sysrow(sysmat, rows, cols, i):
    A = np.memmap(sysmat, dtype=np.float32, mode="r", shape=(rows, cols))
    return np.array(A[i, :], dtype=np.float32)

def nearest_in_layer(det_xyz, idxs, xz):
    x = det_xyz[idxs, 0]; z = det_xyz[idxs, 2]
    tx, tz = xz
    j = np.argmin((x-tx)**2 + (z-tz)**2)
    return int(idxs[j])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sysmat", default="PE_SysMat_shift_0.000000_0.000000_0.000000_0.000000.sysmat".replace("_0.000000_0.000000_0.000000", "_0.000000_0.000000_0.000000"))
    ap.add_argument("--img", default="Params_Image.dat")
    ap.add_argument("--detpar", default="Params_Detector.dat")
    ap.add_argument("--rows", type=int, default=5632)
    ap.add_argument("--cols", type=int, default=25600)
    ap.add_argument("--anchor_det", type=int, default=0, help="detector index used to pick matching (x,z) across layers")
    ap.add_argument("--out", default="mc_ppdf_layers_paperstyle.png")
    ap.add_argument("--percentile", type=float, default=99.7, help="vmax per panel = this percentile (robust)")
    args = ap.parse_args()

    nx, ny, nz, vx, vy, vz = load_img_params(args.img)
    if nx*ny*nz != args.cols:
        raise SystemExit(f"FOV voxels nx*ny*nz={nx*ny*nz} != cols={args.cols}")

    nDet, det = load_det(args.detpar)
    if nDet != args.rows:
        raise SystemExit(f"Detector count mismatch: Params_Detector={nDet}, rows={args.rows}")
    if not (0 <= args.anchor_det < nDet):
        raise SystemExit("Bad --anchor_det")

    # Expected layer layout from your generator
    L123 = 32*16
    L4 = 64*64
    if L123*3 + L4 != nDet:
        raise SystemExit("Detector ordering does not match expected 32x16x3 + 64x64")

    layers = [
        np.arange(0, L123),
        np.arange(L123, 2*L123),
        np.arange(2*L123, 3*L123),
        np.arange(3*L123, 3*L123 + L4),
    ]

    det_xyz = det[:, [0,1,2]]
    ax_xz = (float(det_xyz[args.anchor_det, 0]), float(det_xyz[args.anchor_det, 2]))

    chosen = [nearest_in_layer(det_xyz, L, ax_xz) for L in layers]

    imgs = []
    for d in chosen:
        p = sysrow(args.sysmat, args.rows, args.cols, d).reshape((nx, ny))
        imgs.append(p.T)  # show with origin lower by default using transpose later

    # Figure: tight vertical stack like the paper
    fig = plt.figure(figsize=(4.0, 10.5))
    gs = fig.add_gridspec(4, 1, left=0.23, right=0.86, top=0.93, bottom=0.05, hspace=0.32)

    for i in range(4):
        ax = fig.add_subplot(gs[i, 0])
        data = imgs[i]
        vmax = np.percentile(data, args.percentile)
        im = ax.imshow(data, origin="lower", vmin=0.0, vmax=vmax, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_ylabel(f"Layer {i+1}", rotation=0, labelpad=38, va="center", fontsize=12)

        # small colorbar per panel
        div = make_axes_locatable(ax)
        cax = div.append_axes("right", size="4.5%", pad=0.10)
        cb = fig.colorbar(im, cax=cax)

        # force scientific notation like "1e-6"
        fmt = ScalarFormatter(useMathText=True)
        fmt.set_powerlimits((-2, 2))
        cb.formatter = fmt
        cb.update_ticks()
        cb.ax.tick_params(labelsize=9)

    fig.suptitle("MC\nPPDF", fontsize=18, fontweight="bold")
    fig.savefig(args.out, dpi=300)
    print("WROTE", args.out)
    print("Chosen detectors:", chosen)

if __name__ == "__main__":
    main()
