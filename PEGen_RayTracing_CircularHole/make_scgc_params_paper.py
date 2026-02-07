#!/usr/bin/env python3
import argparse
import math
import random
import numpy as np

# ------------------------------------------------------------
# IMPORTANT: repo reads FIXED counts with fread():
#   Collimator: 80000 floats
#   Detector:   80000 floats
#   Image:        100 floats
#   Physics:      100 floats
# If you write shorter files, you can get uninitialized garbage -> segfaults.
# ------------------------------------------------------------
COL_F32 = 80000
DET_F32 = 80000
IMG_F32 = 100
PHY_F32 = 100

def write_f32_fixed(path: str, arr: np.ndarray, fixed_len: int):
    out = np.zeros(fixed_len, dtype=np.float32)
    arr = np.asarray(arr, dtype=np.float32).ravel()
    if len(arr) > fixed_len:
        raise RuntimeError(f"{path}: need {len(arr)} floats, but fixed_len={fixed_len}")
    out[:len(arr)] = arr
    with open(path, "wb") as f:
        f.write(out.tobytes())

def sample_holes_random_nonoverlap(n_holes, wx, hz, r, seed=0, max_tries=5_000_000):
    """
    Random holes on X-Z face, with light overlap rejection.
    This is enough to match "randomly distributed circular holes" from the paper.
    """
    rng = random.Random(seed)
    holes = []
    min_sep = 2.05 * r  # slightly > diameter to avoid obvious overlap

    tries = 0
    while len(holes) < n_holes and tries < max_tries:
        tries += 1
        x = rng.uniform(-wx/2 + r, wx/2 - r)
        z = rng.uniform(-hz/2 + r, hz/2 - r)

        ok = True
        # check last ~500 holes only (fast)
        for (x2, z2) in holes[-500:]:
            if (x-x2)*(x-x2) + (z-z2)*(z-z2) < (min_sep*min_sep):
                ok = False
                break
        if ok:
            holes.append((x, z))

    return holes

def build_param_image(nx, ny, nz, vx, vy, vz, num_rot, ang_per_rot, sx, sy, sz, fov2coll0):
    img = np.zeros(IMG_F32, dtype=np.float32)
    img[0]  = nx
    img[1]  = ny
    img[2]  = nz
    img[3]  = vx
    img[4]  = vy
    img[5]  = vz
    img[6]  = num_rot
    img[7]  = ang_per_rot
    img[8]  = sx
    img[9]  = sy
    img[10] = sz
    img[11] = fov2coll0
    return img

def build_param_physics(use_compton, save_pe, save_c, save_sum, same_win,
                        e_low, e_high, e_target,
                        calc_crys_rel, calc_coll_rel):
    phy = np.zeros(PHY_F32, dtype=np.float32)
    phy[0] = 1.0 if use_compton else 0.0
    phy[1] = 1.0 if save_pe else 0.0
    phy[2] = 1.0 if save_c else 0.0
    phy[3] = 1.0 if save_sum else 0.0
    phy[4] = 1.0 if same_win else 0.0
    phy[5] = e_low
    phy[6] = e_high
    phy[7] = e_target
    phy[8] = 1.0 if calc_crys_rel else 0.0
    phy[9] = 1.0 if calc_coll_rel else 0.0
    return phy

def build_param_collimator_plate(wx, hz, thick, hole_diam, open_frac,
                                 y0=0.0, seed=0,
                                 mu_tot=1.0, mu_pe=0.8, mu_c=0.2):
    """
    Paper: tungsten coded aperture plate:
      - 12.5% open ratio
      - 1.6 mm diameter random circular holes
      - plate modeled as cuboid with cylindrical air holes
    """
    r = hole_diam / 2.0
    plate_area = wx * hz
    hole_area = math.pi * r * r
    n_holes_target = int(round((open_frac * plate_area) / hole_area))

    # sample random hole centers
    holes = sample_holes_random_nonoverlap(
        n_holes=n_holes_target,
        wx=wx, hz=hz, r=r, seed=seed
    )

    n_holes = len(holes)

    # Param_Collimator format from README:
    # [0] = numLayers
    # layer0 at (0+1)*10 = 10 .. 17
    # holes start at 100 + hid*9
    col = np.zeros(COL_F32, dtype=np.float32)
    col[0] = 1.0  # one layer

    layer0 = 10
    col[layer0 + 0] = float(n_holes)
    col[layer0 + 1] = float(wx)
    col[layer0 + 2] = float(thick)
    col[layer0 + 3] = float(hz)
    col[layer0 + 4] = 0.0  # distance to first layer (0 for first)
    col[layer0 + 5] = float(mu_tot)
    col[layer0 + 6] = float(mu_pe)
    col[layer0 + 7] = float(mu_c)

    y1 = y0 + thick
    base = 100
    needed = base + n_holes * 9
    if needed >= COL_F32:
        raise RuntimeError(f"Too many holes for COL_F32={COL_F32}. Need index {needed}.")

    # Each hole: [x, y0, y1, z, r, mu_tot_hole, mu_pe_hole, mu_c_hole, flag]
    for hid, (x, z) in enumerate(holes):
        off = base + hid * 9
        col[off + 0] = float(x)
        col[off + 1] = float(y0)
        col[off + 2] = float(y1)
        col[off + 3] = float(z)
        col[off + 4] = float(r)
        col[off + 5] = 0.0
        col[off + 6] = 0.0
        col[off + 7] = 0.0
        col[off + 8] = 1.0

    return col, n_holes

def add_detector_layer(det_rows, nx, nz, size_xyz, y_center,
                       mu_tot=1.0, mu_pe=0.8, mu_c=0.2,
                       energy_res=0.20, rot_angle=0.0):
    """
    Detector bins are cuboids arranged in X-Z mosaic plane, centered at y=y_center.

    IMPORTANT MAPPING (to avoid the impossible 64x64 exploding dimension):
    The paper crystal size is given as 3(x)×3(y)×6(z).
    In THIS repo, the axis convention is:
      - width  = X
      - thickness = Y (camera axis, photon travel depth)
      - height = Z

    Therefore we map the 'depth' dimension (6 mm) to Y-thickness.
    That gives:
      layers 1-3: (X=3, Y=6, Z=3)
      layer 4:    (X=2, Y=6, Z=2)

    This matches camera face sizes (64*2=128mm) and avoids nonsense like 64*6=384mm.
    """
    sx, sy, sz = size_xyz
    pitch_x = sx
    pitch_z = sz

    # centered grid in X and Z
    for iz in range(nz):
        z = (iz - (nz - 1) / 2.0) * pitch_z
        for ix in range(nx):
            x = (ix - (nx - 1) / 2.0) * pitch_x
            det_rows.append([
                x, y_center, z,
                sx, sy, sz,
                mu_tot, mu_pe, mu_c,
                energy_res,
                rot_angle,
                1.0
            ])

def build_param_detector_scgc(plate_thick, gap_plate_to_l1, gap_between_layers,
                              l123_nx=32, l123_nz=16,
                              l4_nx=64, l4_nz=64,
                              l123_size=(3.0, 6.0, 3.0),
                              l4_size=(2.0, 6.0, 2.0),
                              mu_tot=1.0, mu_pe=0.8, mu_c=0.2,
                              energy_res=0.20):

    det_rows = []

    # plate spans y in [0, plate_thick]
    y_plate1 = plate_thick

    # layer centers stacked along +Y behind plate
    # NOTE: Y is thickness axis in this repo.
    l1_sy = l123_size[1]
    l4_sy = l4_size[1]

    y_l1 = y_plate1 + gap_plate_to_l1 + l1_sy / 2.0
    y_l2 = y_l1 + l1_sy / 2.0 + gap_between_layers + l1_sy / 2.0
    y_l3 = y_l2 + l1_sy / 2.0 + gap_between_layers + l1_sy / 2.0
    y_l4 = y_l3 + l1_sy / 2.0 + gap_between_layers + l4_sy / 2.0

    add_detector_layer(det_rows, l123_nx, l123_nz, l123_size, y_l1,
                       mu_tot, mu_pe, mu_c, energy_res)
    add_detector_layer(det_rows, l123_nx, l123_nz, l123_size, y_l2,
                       mu_tot, mu_pe, mu_c, energy_res)
    add_detector_layer(det_rows, l123_nx, l123_nz, l123_size, y_l3,
                       mu_tot, mu_pe, mu_c, energy_res)
    add_detector_layer(det_rows, l4_nx, l4_nz, l4_size, y_l4,
                       mu_tot, mu_pe, mu_c, energy_res)

    num_det = len(det_rows)
    needed = 1 + 12 * num_det
    if needed > DET_F32:
        raise RuntimeError(f"Too many detector bins: need {needed} floats but DET_F32={DET_F32}.")

    det = np.zeros(DET_F32, dtype=np.float32)
    det[0] = float(num_det)

    for i, row in enumerate(det_rows):
        base = 1 + i * 12
        det[base:base + 12] = np.array(row, dtype=np.float32)

    return det, num_det

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=".", help="Output folder")
    ap.add_argument("--seed", type=int, default=0)

    # Paper constants
    ap.add_argument("--plate_wx", type=float, default=200.0)
    ap.add_argument("--plate_hz", type=float, default=150.0)
    ap.add_argument("--plate_thick", type=float, default=5.0)   # paper dataset varies 1..10; SCGC exact not stated
    ap.add_argument("--open_frac", type=float, default=0.125)   # 12.5%
    ap.add_argument("--hole_diam", type=float, default=1.6)     # 1.6 mm

    # Layer gaps (paper doesn’t give exact; keep them explicit)
    ap.add_argument("--gap_plate_to_l1", type=float, default=1.0)
    ap.add_argument("--gap_between_layers", type=float, default=1.0)

    # Physics (paper)
    ap.add_argument("--e_low", type=float, default=112.0)
    ap.add_argument("--e_high", type=float, default=168.0)
    ap.add_argument("--e_target", type=float, default=140.0)
    ap.add_argument("--energy_res", type=float, default=0.20)

    # FOV: repo realistically supports 2D plane runs; keep default 2D (nz=1)
    ap.add_argument("--fov_nx", type=int, default=160)
    ap.add_argument("--fov_ny", type=int, default=160)
    ap.add_argument("--fov_nz", type=int, default=1)
    ap.add_argument("--vox", type=float, default=1.0)
    ap.add_argument("--fov2coll0", type=float, default=100.0,
                    help="Distance from FOV CENTER to collimator layer0 (mm). For 2D plane test: set 30, 100, 190, etc.")
    ap.add_argument("--shift_x", type=float, default=0.0)
    ap.add_argument("--shift_y", type=float, default=0.0)
    ap.add_argument("--shift_z", type=float, default=0.0)

    # Save flags
    ap.add_argument("--use_compton", action="store_true")
    ap.add_argument("--save_pe", action="store_true", default=True)
    ap.add_argument("--save_c", action="store_true", default=False)
    ap.add_argument("--save_sum", action="store_true", default=False)

    args = ap.parse_args()

    # -------- Param_Image.dat --------
    img = build_param_image(
        nx=args.fov_nx, ny=args.fov_ny, nz=args.fov_nz,
        vx=args.vox, vy=args.vox, vz=args.vox,
        num_rot=1, ang_per_rot=2 * math.pi,
        sx=args.shift_x, sy=args.shift_y, sz=args.shift_z,
        fov2coll0=args.fov2coll0
    )

    # -------- Param_Physics.dat --------
    phy = build_param_physics(
        use_compton=args.use_compton,
        save_pe=args.save_pe,
        save_c=args.save_c,
        save_sum=args.save_sum,
        same_win=True,
        e_low=args.e_low, e_high=args.e_high, e_target=args.e_target,
        calc_crys_rel=False,
        calc_coll_rel=False
    )

    # -------- Param_Collimator.dat --------
    col, n_holes = build_param_collimator_plate(
        wx=args.plate_wx, hz=args.plate_hz, thick=args.plate_thick,
        hole_diam=args.hole_diam, open_frac=args.open_frac,
        y0=0.0, seed=args.seed,
        mu_tot=1.0, mu_pe=0.8, mu_c=0.2
    )

    # -------- Param_Detector.dat --------
    # Map paper sizes to repo axes (depth=Y thickness)
    # layers 1-3: 3(x) x 6(depth) x 3(z)
    # layer 4:    2(x) x 6(depth) x 2(z)
    det, n_det = build_param_detector_scgc(
        plate_thick=args.plate_thick,
        gap_plate_to_l1=args.gap_plate_to_l1,
        gap_between_layers=args.gap_between_layers,
        l123_nx=32, l123_nz=16,
        l4_nx=64, l4_nz=64,
        l123_size=(3.0, 6.0, 3.0),
        l4_size=(2.0, 6.0, 2.0),
        mu_tot=1.0, mu_pe=0.8, mu_c=0.2,
        energy_res=args.energy_res
    )

    # Write outputs
    out_dir = args.out_dir.rstrip("/")
    write_f32_fixed(f"{out_dir}/Param_Image.dat", img, IMG_F32)
    write_f32_fixed(f"{out_dir}/Param_Physics.dat", phy, PHY_F32)
    write_f32_fixed(f"{out_dir}/Param_Collimator.dat", col, COL_F32)
    write_f32_fixed(f"{out_dir}/Param_Detector.dat", det, DET_F32)

    print("Wrote:")
    print(f"  {out_dir}/Param_Image.dat   (IMG_F32={IMG_F32})")
    print(f"  {out_dir}/Param_Physics.dat (PHY_F32={PHY_F32})")
    print(f"  {out_dir}/Param_Collimator.dat (holes={n_holes}, COL_F32={COL_F32})")
    print(f"  {out_dir}/Param_Detector.dat   (detectors={n_det}, DET_F32={DET_F32})")
    print()
    print("SCGC paper checks:")
    print(f"  plate: {args.plate_wx} x {args.plate_hz} mm, open={args.open_frac*100:.1f}%, hole_diam={args.hole_diam} mm")
    print("  layer1-3: 32x16 crystals, size mapped to (3,6,3) with depth=Y")
    print("  layer4:   64x64 crystals, size mapped to (2,6,2) with depth=Y")
    print()
    print("NOTE:")
    print("  Micro-units are internal to GPU code, not in Param_Detector.dat.")
    print("  For 3D FOV (160^3) the sysmat is enormous; the repo examples typically validate 2D FOV planes.")

if __name__ == "__main__":
    main()
