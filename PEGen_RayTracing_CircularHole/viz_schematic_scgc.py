#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def cuboid_faces(center, size):
    cx, cy, cz = center
    sx, sy, sz = size
    x = [cx - sx/2, cx + sx/2]
    y = [cy - sy/2, cy + sy/2]
    z = [cz - sz/2, cz + sz/2]
    # 8 vertices
    v = np.array([
        [x[0], y[0], z[0]],
        [x[1], y[0], z[0]],
        [x[1], y[1], z[0]],
        [x[0], y[1], z[0]],
        [x[0], y[0], z[1]],
        [x[1], y[0], z[1]],
        [x[1], y[1], z[1]],
        [x[0], y[1], z[1]],
    ])
    faces = [
        [v[0], v[1], v[2], v[3]],  # bottom
        [v[4], v[5], v[6], v[7]],  # top
        [v[0], v[1], v[5], v[4]],  # front
        [v[2], v[3], v[7], v[6]],  # back
        [v[1], v[2], v[6], v[5]],  # right
        [v[3], v[0], v[4], v[7]],  # left
    ]
    return faces

def load_detector_binary(path):
    a = np.fromfile(path, dtype=np.float32)
    n = int(round(float(a[0])))
    det = a[1:].reshape(n, 12)
    return det

def load_collimator_holes(path):
    # Your generator writes holes at 100 + hid*9:
    col = np.fromfile(path, dtype=np.float32)
    num_holes = int(round(col[10]))  # layer0+0
    holes = []
    for hid in range(num_holes):
        off = 100 + hid*9
        x = float(col[off+0])
        z = float(col[off+3])
        r = float(col[off+4])
        holes.append((x, z, r))
    # plate dims from layer0 block
    W = float(col[11])  # layer0+1
    T = float(col[12])  # layer0+2
    H = float(col[13])  # layer0+3
    return W, T, H, holes

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detpar", default="Params_Detector.dat")
    ap.add_argument("--colpar", default="Params_Collimator.dat")
    ap.add_argument("--out", default="scgc_schematic.png")
    ap.add_argument("--max_det_draw", type=int, default=200, help="draw only this many detector blocks per layer for clarity")
    args = ap.parse_args()

    det = load_detector_binary(args.detpar)
    W, T, H, holes = load_collimator_holes(args.colpar)

    # Split into layers using unique y
    ys = det[:,1]
    y_levels = np.unique(np.round(ys, 4))
    y_levels.sort()

    # Figure layout: left 3D schematic, right 3 slice panels
    fig = plt.figure(figsize=(10.5, 7.8))
    ax3d = fig.add_axes([0.05, 0.08, 0.58, 0.84], projection="3d")

    # Draw coded-aperture plate as a thin slab
    plate_center = (0.0, T/2.0, 0.0)
    plate_size = (W, T, H)
    faces = cuboid_faces(plate_center, plate_size)
    plate = Poly3DCollection(faces, alpha=0.35, linewidths=0.3)
    ax3d.add_collection3d(plate)

    # Draw holes as circles projected on plate top face (visual only)
    # We'll draw just a subset for speed/clarity
    hole_subset = holes[::max(1, len(holes)//60)]
    for (x,z,r) in hole_subset:
        # represent hole as a short "tube" outline using a small ring at top
        theta = np.linspace(0, 2*np.pi, 40)
        xx = x + r*np.cos(theta)
        zz = z + r*np.sin(theta)
        yy = np.full_like(xx, T)  # top surface
        ax3d.plot(xx, yy, zz, linewidth=0.6)

    # Draw detector blocks: sample subset per layer
    for li, y0 in enumerate(y_levels):
        layer_idx = np.where(np.abs(ys - y0) < 1e-3)[0]
        # subsample to avoid 5632 cubes
        if layer_idx.size > args.max_det_draw:
            layer_idx = layer_idx[::layer_idx.size // args.max_det_draw]
        for idx in layer_idx:
            x,y,z = det[idx,0], det[idx,1], det[idx,2]
            sx,sy,sz = det[idx,3], det[idx,4], det[idx,5]
            faces = cuboid_faces((x,y,z), (sx,sy,sz))
            cube = Poly3DCollection(faces, alpha=0.25, linewidths=0.2)
            ax3d.add_collection3d(cube)

    # Rays: simple illustrative beams through plate down to deeper layers
    # Draw 3 rays
    ray_xz = [( -20,  10), (0,0), ( 25, -15)]
    for (rx, rz) in ray_xz:
        ax3d.plot([rx, rx], [T, float(y_levels[-1])], [rz, rz], linewidth=2.0)

    # 3D axis styling
    ax3d.set_xlabel("X (mm)")
    ax3d.set_ylabel("Y (mm)")
    ax3d.set_zlabel("Z (mm)")
    ax3d.set_title("Coded-Aperture Plate + Detector Layers (schematic)")
    ax3d.view_init(elev=22, azim=-55)
    ax3d.set_xlim(-W/2, W/2)
    ax3d.set_ylim(-5, float(y_levels[-1]) + 10)
    ax3d.set_zlim(-H/2, H/2)
    ax3d.grid(False)

    # Right-side slice panels (stylized shapes like your example)
    # We’ll draw 3 panels: Slice A, B, C
    def slice_panel(rect, label, scale=1.0):
        ax = fig.add_axes(rect)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.add_patch(Rectangle((0,0), 1, 1, fill=False, linestyle="--", linewidth=1))
        # two crescents + center square (simple approximation)
        # left crescent
        ax.add_patch(Circle((0.32, 0.62), 0.12*scale, color="#f3b37a", alpha=0.85))
        ax.add_patch(Circle((0.35, 0.62), 0.09*scale, color="white", alpha=1.0))
        # right crescent
        ax.add_patch(Circle((0.68, 0.62), 0.12*scale, color="#f3b37a", alpha=0.85))
        ax.add_patch(Circle((0.65, 0.62), 0.09*scale, color="white", alpha=1.0))
        # center square
        s = 0.08*scale
        ax.add_patch(Rectangle((0.5 - s/2, 0.45 - s/2), s, s, color="#f3b37a", alpha=0.85))
        ax.set_title(label, fontsize=11)
        return ax

    # top/mid/bottom slice panels
    slice_panel([0.70, 0.66, 0.26, 0.25], "Slice A", scale=1.0)
    slice_panel([0.70, 0.38, 0.26, 0.25], "Slice B", scale=0.85)
    slice_panel([0.70, 0.10, 0.26, 0.25], "Slice C", scale=0.65)

    # Left labels to match your figure
    fig.text(0.06, 0.74, "Coded-Aperture\nPlate", fontsize=10, va="center")
    fig.text(0.06, 0.52, "Self-Collimating\nDetectors", fontsize=10, va="center")
    fig.text(0.06, 0.23, "Panel Detectors", fontsize=10, va="center")

    fig.savefig(args.out, dpi=250)
    print("WROTE", args.out)

if __name__ == "__main__":
    main()
