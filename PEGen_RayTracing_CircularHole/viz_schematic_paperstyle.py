#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch, Polygon
from matplotlib.collections import PatchCollection

def f32(path): return np.fromfile(path, dtype=np.float32)

def load_det(path):
    a = f32(path)
    n = int(round(float(a[0])))
    det = a[1:].reshape(n, 12)
    return det

def load_plate(path):
    col = f32(path)
    # from your generator: layer0 at offset 10
    W = float(col[11])  # width X
    T = float(col[12])  # thickness Y
    H = float(col[13])  # height Z
    num_holes = int(round(float(col[10])))
    holes = []
    for hid in range(num_holes):
        off = 100 + hid*9
        x = float(col[off+0]); z = float(col[off+3]); r = float(col[off+4])
        holes.append((x,z,r))
    return W,T,H,holes

def draw_slice(ax, title, scale):
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.add_patch(Rectangle((0,0), 1, 1, fill=False, linestyle="--", linewidth=1))
    col = "#f3b37a"
    # crescents
    ax.add_patch(Circle((0.32, 0.62), 0.13*scale, color=col, alpha=0.85))
    ax.add_patch(Circle((0.35, 0.62), 0.10*scale, color="white"))
    ax.add_patch(Circle((0.68, 0.62), 0.13*scale, color=col, alpha=0.85))
    ax.add_patch(Circle((0.65, 0.62), 0.10*scale, color="white"))
    # square
    s = 0.10*scale
    ax.add_patch(Rectangle((0.5-s/2, 0.45-s/2), s, s, color=col, alpha=0.85))
    ax.set_title(title, fontsize=12)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detpar", default="Params_Detector.dat")
    ap.add_argument("--colpar", default="Params_Collimator.dat")
    ap.add_argument("--out", default="schematic_paperstyle.png")
    args = ap.parse_args()

    det = load_det(args.detpar)
    W,T,H,holes = load_plate(args.colpar)

    # Identify unique layer y positions
    ys = det[:,1]
    y_levels = np.unique(np.round(ys, 4))
    y_levels.sort()

    fig = plt.figure(figsize=(10.5, 7.0))

    # Left schematic canvas (2D pseudo-3D)
    ax = fig.add_axes([0.05, 0.08, 0.62, 0.84])
    ax.set_axis_off()
    ax.set_title("Coded-Aperture Plate + Self-Collimating Detectors + Panel Detectors", fontsize=14, pad=10)

    # Perspective parameters
    # We'll draw in 2D with a fake depth shift
    dxp, dyp = 0.18, 0.10  # depth offset in axes coords

    # Plate (top)
    plate = Polygon([[0.15,0.78],[0.65,0.90],[0.85,0.82],[0.35,0.70]], closed=True,
                    facecolor="#bbbbbb", edgecolor="#888888", alpha=0.9)
    ax.add_patch(plate)

    # Holes (subset)
    holes = holes[::max(1, len(holes)//8)]
    for (x,z,r) in holes:
        # place holes roughly on plate
        cx = 0.25 + 0.5*(x/(W/2))*0.35
        cy = 0.80 + 0.5*(z/(H/2))*0.12
        ax.add_patch(Circle((cx, cy), 0.018, color="white", alpha=0.95))

    ax.text(0.02, 0.78, "Coded-Aperture\nPlate", fontsize=12, va="center")

    # Detector layers as stacked grids of small blocks
    # We’ll draw 3 self-collimating layers and one panel detector layer.
    def draw_layer(y0, ybase, nblocks_x, nblocks_y, label=None):
        patches = []
        # grid on a tilted plane
        x0, y0a = 0.22, ybase
        w, h = 0.55, 0.18
        for iy in range(nblocks_y):
            for ix in range(nblocks_x):
                px = x0 + (ix/(nblocks_x))*w + (iy/(nblocks_y))*dxp*0.25
                py = y0a + (iy/(nblocks_y))*h - (ix/(nblocks_x))*dyp*0.15
                patches.append(FancyBboxPatch((px, py), 0.018, 0.028,
                                             boxstyle="round,pad=0.002,rounding_size=0.002",
                                             linewidth=0.2, edgecolor="#4c4c4c",
                                             facecolor="#97b6c8", alpha=0.9))
        pc = PatchCollection(patches, match_original=True)
        ax.add_collection(pc)
        if label:
            ax.text(0.02, ybase+0.08, label, fontsize=12, va="center")

    draw_layer(y_levels[0], 0.56, 10, 6, label="Self-Collimating\nDetectors")
    draw_layer(y_levels[1], 0.44, 10, 6)
    draw_layer(y_levels[2], 0.32, 10, 6)

    # Panel detectors (big slab)
    ax.add_patch(Polygon([[0.22,0.18],[0.72,0.30],[0.86,0.23],[0.36,0.11]], closed=True,
                         facecolor="#9fb9a5", edgecolor="#6f8a76", alpha=0.95))
    ax.text(0.02, 0.20, "Panel Detectors", fontsize=12, va="center")

    # Rays (three beams)
    for x in [0.40, 0.53, 0.60]:
        ax.add_patch(Polygon([[x,0.86],[x+0.03,0.86],[x+0.08,0.64],[x+0.05,0.64]],
                             closed=True, facecolor="#f3b37a", edgecolor="none", alpha=0.65))
        ax.add_patch(Polygon([[x+0.06,0.64],[x+0.09,0.64],[x+0.12,0.48],[x+0.09,0.48]],
                             closed=True, facecolor="#f3b37a", edgecolor="none", alpha=0.65))

    # Right slice panels
    axA = fig.add_axes([0.72, 0.66, 0.25, 0.25]); draw_slice(axA, "Slice A", 1.0)
    axB = fig.add_axes([0.72, 0.38, 0.25, 0.25]); draw_slice(axB, "Slice B", 0.85)
    axC = fig.add_axes([0.72, 0.10, 0.25, 0.25]); draw_slice(axC, "Slice C", 0.65)

    fig.savefig(args.out, dpi=300)
    print("WROTE", args.out)

if __name__ == "__main__":
    main()
