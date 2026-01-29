#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def read_f32(path):
    return np.fromfile(path, dtype=np.float32)

def load_image_params(path):
    a = read_f32(path)
    if a.size < 6:
        raise RuntimeError(f"{path} too small")
    nx, ny, nz = int(a[0]), int(a[1]), int(a[2]) if a.size >= 3 else 1
    vx, vy, vz = float(a[3]), float(a[4]), float(a[5])
    return nx, ny, nz, vx, vy, vz

def load_detector_params(path):
    a = read_f32(path)
    if a.size < 1:
        raise RuntimeError(f"{path} empty")
    numDet = int(round(float(a[0])))
    expected = 1 + 12*numDet
    if a.size != expected:
        raise RuntimeError(f"{path}: got {a.size} floats, expected {expected} (=1+12*numDet), numDet={numDet}")
    det = a[1:].reshape(numDet, 12)
    # columns: x,y,z,sx,sy,sz,mu_tot,mu_pe,mu_c,energyRes,rotAngle,flag
    x,y,z = det[:,0], det[:,1], det[:,2]
    sx,sy,sz = det[:,3], det[:,4], det[:,5]
    return numDet, det, x,y,z,sx,sy,sz

def memmap_sysmat_row(sysmat_path, rows, cols, det_idx):
    A = np.memmap(sysmat_path, dtype=np.float32, mode="r", shape=(rows, cols))
    return np.array(A[det_idx, :], dtype=np.float32)

def layer_ids_from_y(y, tol=1e-4):
    # group nearly-equal y values into layers
    ys = np.sort(np.unique(np.round(y / tol) * tol))
    # map each detector y to nearest layer
    layer = np.zeros_like(y, dtype=int)
    for i, yi in enumerate(y):
        layer[i] = int(np.argmin(np.abs(ys - yi)))
    return ys, layer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sysmat", default="PE_SysMat_shift_0.000000_0.000000_0.000000.sysmat")
    ap.add_argument("--image_params", default="Params_Image.dat")
    ap.add_argument("--detector_params", default="Params_Detector.dat")
    ap.add_argument("--rows", type=int, default=5632)
    ap.add_argument("--cols", type=int, default=25600)
    ap.add_argument("--det", type=int, default=0, help="detector index to highlight")
    ap.add_argument("--out", default="nice_overlay.png")
    ap.add_argument("--log_floor", type=float, default=-40.0, help="lower clamp for log10(PPDF)")
    ap.add_argument("--zoom_mm", type=float, default=60.0, help="zoom inset width/height in mm (centered at 0,0)")
    ap.add_argument("--det_alpha", type=float, default=0.35)
    ap.add_argument("--ppdf_alpha", type=float, default=0.85)
    args = ap.parse_args()

    nx, ny, nz, vx, vy, vz = load_image_params(args.image_params)
    fov_vox = nx*ny*nz
    if fov_vox != args.cols:
        raise RuntimeError(f"FOV voxels={fov_vox} but cols={args.cols}. Fix inputs.")

    numDet, det, dx, dy, dz, dsx, dsy, dsz = load_detector_params(args.detector_params)
    if numDet != args.rows:
        raise RuntimeError(f"Detector count mismatch: file says {numDet}, args.rows={args.rows}")
    if not (0 <= args.det < numDet):
        raise ValueError(f"--det must be in [0..{numDet-1}]")

    # Load PPDF for selected detector and reshape to (nx,ny) (your case 160x160)
    row = memmap_sysmat_row(args.sysmat, args.rows, args.cols, args.det)
    p = row.reshape((nx, ny))

    # log10 with epsilon, clamp for visualization
    eps = np.float32(1e-12)
    p_log = np.log10(np.maximum(p, eps))
    p_log = np.clip(p_log, args.log_floor, np.max(p_log))

    # FOV physical extent (x and "z" since your FOV is 160x160x1)
    x_min = -(nx-1)/2 * vx
    x_max =  (nx-1)/2 * vx
    z_min = -(ny-1)/2 * vz
    z_max =  (ny-1)/2 * vz

    # Layer grouping for detector coloring
    layer_y, layer_id = layer_ids_from_y(dy)

    # ---- Figure ----
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(1,1,1)
    ax.set_aspect("equal")

    # 1) PPDF overlay (behind geometry)
    im = ax.imshow(
        p_log.T,
        origin="lower",
        extent=[x_min, x_max, z_min, z_max],
        aspect="equal",
        alpha=args.ppdf_alpha
    )

    # 2) Draw FOV box
    ax.add_patch(Rectangle((x_min, z_min), x_max-x_min, z_max-z_min,
                           fill=False, linewidth=2))

    # 3) Detector geometry rectangles in x–z plane
    # Use different edge colors per layer (cycle through matplotlib default color cycle)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i in range(numDet):
        c = colors[layer_id[i] % len(colors)]
        # rectangle centered at (dx, dz), width=dsx, height=dsz
        r = Rectangle((dx[i] - dsx[i]/2, dz[i] - dsz[i]/2),
                      dsx[i], dsz[i],
                      fill=False, linewidth=1.0, edgecolor=c, alpha=args.det_alpha)
        ax.add_patch(r)

    # 4) Highlight selected detector in red, thicker
    i = args.det
    rsel = Rectangle((dx[i] - dsx[i]/2, dz[i] - dsz[i]/2),
                     dsx[i], dsz[i],
                     fill=False, linewidth=3.0, edgecolor="red", alpha=1.0)
    ax.add_patch(rsel)

    # Title/labels
    ax.set_title(f"Detector geometry (x–z) + PPDF overlay (det={args.det})")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("z (mm)")
    ax.grid(alpha=0.25)

    # Colorbar
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("log10(PPDF)")

    # 5) Inset zoom (bottom-left)
    zoom = args.zoom_mm
    axins = inset_axes(ax, width="35%", height="35%", loc="lower left", borderpad=2)
    axins.imshow(
        p_log.T,
        origin="lower",
        extent=[x_min, x_max, z_min, z_max],
        aspect="equal",
        alpha=1.0
    )
    axins.set_xlim(-zoom/2, zoom/2)
    axins.set_ylim(-zoom/2, zoom/2)
    axins.set_title("Zoom: FOV", fontsize=10)
    axins.grid(alpha=0.2)

    # Optional: draw the FOV box in inset too (cleaner reference)
    axins.add_patch(Rectangle((x_min, z_min), x_max-x_min, z_max-z_min,
                              fill=False, linewidth=2))

    fig.tight_layout()
    fig.savefig(args.out, dpi=220)
    print(f"WROTE {args.out}")

if __name__ == "__main__":
    main()
