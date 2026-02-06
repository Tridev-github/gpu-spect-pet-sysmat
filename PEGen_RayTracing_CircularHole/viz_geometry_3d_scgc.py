#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def read_f32(path):
    return np.fromfile(path, dtype=np.float32)

def cuboid_faces(cx, cy, cz, sx, sy, sz):
    # center + sizes -> 8 corners
    x0, x1 = cx - sx/2, cx + sx/2
    y0, y1 = cy - sy/2, cy + sy/2
    z0, z1 = cz - sz/2, cz + sz/2
    v = np.array([
        [x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],
        [x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1],
    ])
    faces = [
        [v[0],v[1],v[2],v[3]],  # bottom z0
        [v[4],v[5],v[6],v[7]],  # top z1
        [v[0],v[1],v[5],v[4]],
        [v[2],v[3],v[7],v[6]],
        [v[1],v[2],v[6],v[5]],
        [v[0],v[3],v[7],v[4]],
    ]
    return faces

def add_cuboid(ax, cx, cy, cz, sx, sy, sz, alpha=0.15, linewidth=0.2):
    faces = cuboid_faces(cx, cy, cz, sx, sy, sz)
    pc = Poly3DCollection(faces, alpha=alpha, linewidths=linewidth, edgecolor="k")
    ax.add_collection3d(pc)

def add_cylinder(ax, cx, cz, y1, y2, r, n=24, alpha=0.15):
    # cylinder axis is Y, centered at (cx,cz)
    theta = np.linspace(0, 2*np.pi, n)
    xs = cx + r*np.cos(theta)
    zs = cz + r*np.sin(theta)
    y = np.array([y1, y2])

    X, Y = np.meshgrid(xs, y)
    Z, _ = np.meshgrid(zs, y)

    ax.plot_surface(X, Y, Z, alpha=alpha, linewidth=0, antialiased=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--param_collimator", default="Param_Collimator.dat")
    ap.add_argument("--param_detector", default="Param_Detector.dat")
    ap.add_argument("--out", default="geometry_scgc_3d.png")
    ap.add_argument("--max_holes", type=int, default=200, help="render at most N holes as cylinders (speed)")
    ap.add_argument("--max_det", type=int, default=500, help="render at most N detector crystals as cuboids (speed)")
    args = ap.parse_args()

    col = read_f32(args.param_collimator)
    det = read_f32(args.param_detector)

    num_holes = int(col[10])  # layer0 at 10
    plate_w = float(col[11])
    plate_t = float(col[12])
    plate_h = float(col[13])

    # plate at y in [0, plate_t] => center y = plate_t/2
    plate_cx, plate_cy, plate_cz = 0.0, plate_t/2, 0.0
    plate_sx, plate_sy, plate_sz = plate_w, plate_t, plate_h

    num_det = int(det[0])

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Plate cuboid
    add_cuboid(ax, plate_cx, plate_cy, plate_cz, plate_sx, plate_sy, plate_sz, alpha=0.10)

    # Holes (cylinders)
    hole_count = min(num_holes, args.max_holes)
    if hole_count > 0:
        idx = np.linspace(0, num_holes - 1, hole_count).astype(int)
        for hid in idx:
            off = 100 + hid*9
            x = float(col[off+0])
            y1 = float(col[off+1])
            y2 = float(col[off+2])
            z = float(col[off+3])
            r = float(col[off+4])
            add_cylinder(ax, x, z, y1, y2, r, alpha=0.10)

    # Detectors (cuboids)
    det_count = min(num_det, args.max_det)
    idx = np.linspace(0, num_det - 1, det_count).astype(int)
    for i in idx:
        base = 1 + i*12
        x = float(det[base+0])
        y = float(det[base+1])
        z = float(det[base+2])
        sx = float(det[base+3])
        sy = float(det[base+4])
        sz = float(det[base+5])
        add_cuboid(ax, x, y, z, sx, sy, sz, alpha=0.08)

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title(f"SCGC Geometry (plate + {hole_count}/{num_holes} holes + {det_count}/{num_det} crystals)")
    ax.view_init(elev=18, azim=35)

    # Reasonable bounds
    ax.set_xlim(-plate_w/2, plate_w/2)
    ax.set_zlim(-plate_h/2, plate_h/2)
    ax.set_ylim(-50, 150)

    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()
