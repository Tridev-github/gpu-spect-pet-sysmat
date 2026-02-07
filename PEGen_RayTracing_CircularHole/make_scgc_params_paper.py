#!/usr/bin/env python3
import numpy as np
import math
import random
import argparse

def write_f32(path: str, arr: np.ndarray) -> None:
    arr = np.asarray(arr, dtype=np.float32)
    arr.tofile(path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_prefix", default="Params", help="Writes <prefix>_Collimator.dat etc.")
    ap.add_argument("--fov2coll", type=float, default=30.0, help="Distance from FOV plane to plate front (mm)")
    ap.add_argument("--plate_thick", type=float, default=5.0, help="Plate thickness (mm) (paper range 1..10)")
    ap.add_argument("--open_frac", type=float, default=0.125, help="Open fraction (paper: 0.125)")
    ap.add_argument("--hole_diam", type=float, default=1.6, help="Hole diameter (mm)")
    ap.add_argument("--seed", type=int, default=0)
    # Gaps are NOT given in pasted text; keep them small/plausible and tweak if you later extract Fig.1 dims.
    ap.add_argument("--gap_plate_l1", type=float, default=2.0)
    ap.add_argument("--gap_l1_l2", type=float, default=2.0)
    ap.add_argument("--gap_l2_l3", type=float, default=2.0)
    ap.add_argument("--gap_l3_l4", type=float, default=2.0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # -----------------------------
    # Paper geometry (from your text)
    # System size: 200 x 150 x 150 mm (X, Y(depth), Z)
    # Plate: 200 x 150 with random holes, open fraction 12.5%, hole diam 1.6mm
    # Detectors: L1-3: 32x16 mosaic; L4: 64x64 mosaic
    # IMPORTANT: Map paper dims to repo params:
    #   Repo expects (widthX, thicknessY, heightZ)
    #   Depth stacking axis is Y.
    # So: "6mm" is depth => thicknessY = 6.
    # In-plane tile is X-Z => (widthX,heightZ) are 3x3 or 2x2.
    # -----------------------------
    PLATE_WX = 200.0
    PLATE_HZ = 150.0
    PLATE_TH = float(args.plate_thick)

    HOLE_DIAM = float(args.hole_diam)
    HOLE_R = HOLE_DIAM / 2.0
    OPEN_FRAC = float(args.open_frac)

    # Detectors (mapped)
    L123_NX, L123_NZ = 32, 16
    L123_WX, L123_TY, L123_HZ = 3.0, 6.0, 3.0  # (X, Y(depth), Z)
    L4_NX, L4_NZ = 64, 64
    L4_WX, L4_TY, L4_HZ = 2.0, 6.0, 2.0

    # 2D evaluation plane: 160x160 in X-Z, single voxel in Y
    NX, NY, NZ = 160, 1, 160
    VOX_X, VOX_Y, VOX_Z = 1.0, 1.0, 1.0

    # Physics (paper)
    E_LOW, E_HIGH = 112.0, 168.0
    E_TARGET = 140.0
    ENERGY_RES = 0.20

    # Coordinate convention used here:
    # Plate spans y in [0, PLATE_TH]
    # Detectors at y > PLATE_TH
    # FOV plane centered at y = -fov2coll (in front of plate)
    PLATE_Y0 = 0.0
    PLATE_Y1 = PLATE_TH

    fov_center = np.array([0.0, -float(args.fov2coll), 0.0], dtype=np.float32)

    # -----------------------------
    # Params_Image.dat (exact 12 floats)
    # -----------------------------
    img = np.array([
        NX, NY, NZ,
        VOX_X, VOX_Y, VOX_Z,
        1, 2*math.pi,
        0.0, 0.0, 0.0,
        float(args.fov2coll)
    ], dtype=np.float32)
    write_f32(f"{args.out_prefix}_Image.dat", img)

    # -----------------------------
    # Params_Physics.dat (exact 10 floats)
    # -----------------------------
    phy = np.array([
        0,      # flagUsingCompton
        1,      # flagSavingPESysmat
        0,      # flagSavingComptonSysmat
        0,      # flagSavingPEComptonSysmat
        1,      # flagUsingSameEnergyWindow
        E_LOW, E_HIGH,
        E_TARGET,
        0, 0    # geo relationship flags (optional)
    ], dtype=np.float32)
    write_f32(f"{args.out_prefix}_Physics.dat", phy)

    # -----------------------------
    # Params_Collimator.dat (dynamic length)
    # -----------------------------
    plate_area = PLATE_WX * PLATE_HZ
    hole_area = math.pi * HOLE_R * HOLE_R
    num_holes_target = int(round((OPEN_FRAC * plate_area) / hole_area))

    # Random holes (rejection spacing, not perfect but prevents dumb overlaps)
    holes = []
    max_tries = 2_000_000
    min_sep = HOLE_DIAM * 1.05
    tries = 0
    while len(holes) < num_holes_target and tries < max_tries:
        tries += 1
        x = random.uniform(-PLATE_WX/2 + HOLE_R, PLATE_WX/2 - HOLE_R)
        z = random.uniform(-PLATE_HZ/2 + HOLE_R, PLATE_HZ/2 - HOLE_R)
        ok = True
        for (x2, z2) in holes[-200:]:
            if (x-x2)**2 + (z-z2)**2 < (min_sep**2):
                ok = False
                break
        if ok:
            holes.append((x, z))

    num_holes = len(holes)
    ncol = 100 + num_holes*9
    col = np.zeros(ncol, dtype=np.float32)

    col[0] = 1  # one layer
    base = 10
    col[base+0] = num_holes
    col[base+1] = PLATE_WX
    col[base+2] = PLATE_TH
    col[base+3] = PLATE_HZ
    col[base+4] = 0.0  # dist from layer0 to itself

    # attenuation placeholders (you can replace with tungsten @140keV)
    col[base+5] = 1.0
    col[base+6] = 0.8
    col[base+7] = 0.2

    for hid, (x,z) in enumerate(holes):
        off = 100 + hid*9
        col[off+0] = x
        col[off+1] = PLATE_Y0
        col[off+2] = PLATE_Y1
        col[off+3] = z
        col[off+4] = HOLE_R
        col[off+5] = 0.0
        col[off+6] = 0.0
        col[off+7] = 0.0
        col[off+8] = 1.0

    write_f32(f"{args.out_prefix}_Collimator.dat", col)

    # -----------------------------
    # Params_Detector.dat (exact 1 + 12*numDet floats)
    # -----------------------------
    det_rows = []

    def add_layer(nx, nz, wx, ty, hz, y_center):
        pitch_x = wx
        pitch_z = hz
        for iz in range(nz):
            for ix in range(nx):
                x = (ix - (nx-1)/2) * pitch_x
                z = (iz - (nz-1)/2) * pitch_z
                det_rows.append([
                    x, y_center, z,
                    wx, ty, hz,
                    1.0, 0.8, 0.2,
                    ENERGY_RES,
                    0.0,
                    1.0
                ])

    y_l1 = PLATE_TH + args.gap_plate_l1 + L123_TY/2
    y_l2 = y_l1 + L123_TY/2 + args.gap_l1_l2 + L123_TY/2
    y_l3 = y_l2 + L123_TY/2 + args.gap_l2_l3 + L123_TY/2
    y_l4 = y_l3 + L123_TY/2 + args.gap_l3_l4 + L4_TY/2

    add_layer(L123_NX, L123_NZ, L123_WX, L123_TY, L123_HZ, y_l1)
    add_layer(L123_NX, L123_NZ, L123_WX, L123_TY, L123_HZ, y_l2)
    add_layer(L123_NX, L123_NZ, L123_WX, L123_TY, L123_HZ, y_l3)
    add_layer(L4_NX,   L4_NZ,   L4_WX,   L4_TY,   L4_HZ,   y_l4)

    num_det = len(det_rows)
    det = np.zeros(1 + 12*num_det, dtype=np.float32)
    det[0] = num_det
    for i, row in enumerate(det_rows):
        det[1+i*12 : 1+(i+1)*12] = np.array(row, dtype=np.float32)

    write_f32(f"{args.out_prefix}_Detector.dat", det)

    print("WROTE:")
    print(f"  {args.out_prefix}_Image.dat     (12 floats)")
    print(f"  {args.out_prefix}_Physics.dat   (10 floats)")
    print(f"  {args.out_prefix}_Collimator.dat ({det.size}?? dynamic)")
    print(f"  {args.out_prefix}_Detector.dat  ({1+12*num_det} floats)")
    print(f"numHoles={num_holes}  numDet={num_det}")
    print("NOTE: 2D FOV is X-Z plane at y=-fov2coll with NY=1.")

if __name__ == "__main__":
    main()
