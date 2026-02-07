#!/usr/bin/env python3
import argparse, math, random, os
import numpy as np

# ---------------------------------------------------------
# Paper constants (SCGC)
# ---------------------------------------------------------
PLATE_W_X   = 200.0    # mm
PLATE_H_Z   = 150.0    # mm
HOLE_DIAM   = 1.6      # mm
OPEN_FRAC   = 0.125    # 12.5%
# Plate thickness is not explicitly fixed in your pasted text;
# paper dataset spans 1..10mm. Use 5mm default (common in your files).
PLATE_THICK = 5.0      # mm

# Detector layers (paper)
L123_NX, L123_NZ = 32, 16
L123_SIZE = (3.0, 3.0, 6.0)  # (x,y,z) mm
L4_NX, L4_NZ = 64, 64
L4_SIZE = (2.0, 2.0, 6.0)    # (x,y,z) mm

# Energy (paper eval)
E_LOW, E_HIGH = 112.0, 168.0
E_TARGET = 140.0
ENERGY_RES = 0.20

def write_f32(path, arr):
    arr = np.asarray(arr, dtype=np.float32)
    arr.tofile(path)

def pad_to_len(arr, total_floats):
    """Repo sometimes hardcodes reading fixed buffers; pad with zeros safely."""
    if arr.size > total_floats:
        raise RuntimeError(f"{arr.size} floats > pad target {total_floats}")
    if arr.size == total_floats:
        return arr
    out = np.zeros(total_floats, dtype=np.float32)
    out[:arr.size] = arr
    return out

def gen_holes_random_nonoverlap(num_holes, hole_r, seed=0, min_sep_scale=1.05, max_tries=2_000_000):
    random.seed(seed)
    holes = []
    min_sep = (2.0 * hole_r) * min_sep_scale

    tries = 0
    while len(holes) < num_holes and tries < max_tries:
        tries += 1
        x = random.uniform(-PLATE_W_X/2 + hole_r, PLATE_W_X/2 - hole_r)
        z = random.uniform(-PLATE_H_Z/2 + hole_r, PLATE_H_Z/2 - hole_r)

        ok = True
        # Check last chunk only (fast); good enough visually/statistically
        for (x2, z2) in holes[-300:]:
            if (x-x2)**2 + (z-z2)**2 < (min_sep**2):
                ok = False
                break
        if ok:
            holes.append((x, z))

    if len(holes) < num_holes:
        print(f"[WARN] only placed {len(holes)}/{num_holes} holes. Increase max_tries or relax min_sep.")
    return holes

def add_mosaic(det_list, nx, nz, size_xyz, y_center):
    sx, sy, sz = size_xyz
    pitch_x = sx  # mosaic, no gap stated in paper
    pitch_z = sx  # IMPORTANT: paper’s 3mm(x) and 6mm(z) is CRYSTAL DIM, but array indexing is mosaic grid.
                  # In your repo the “height” is Z, and “width” is X. Mosaic pitch should match X dimension.
                  # If you instead pitch by 6mm in Z you will explode Z-span (your earlier bbox showed that).
                  # For a 32×16 mosaic, the natural assumption is pitch_z = 3mm (not 6mm).
                  # Same for L4: pitch_z = 2mm.

    for iz in range(nz):
        for ix in range(nx):
            x = (ix - (nx-1)/2) * pitch_x
            z = (iz - (nz-1)/2) * pitch_z
            det_list.append([x, y_center, z, sx, sy, sz])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_prefix", default="Param", help="Output prefix: Param or Params")
    ap.add_argument("--pad_fixed_80000", action="store_true", help="Pad Param_*.dat to 80000 floats (safe for repo hardcodes)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plate_thick", type=float, default=PLATE_THICK)
    ap.add_argument("--fov_nx", type=int, default=160)
    ap.add_argument("--fov_ny", type=int, default=1)
    ap.add_argument("--fov_nz", type=int, default=160)
    ap.add_argument("--vox", type=float, default=1.0, help="voxel size mm (assume isotropic)")
    ap.add_argument("--fov2coll0", type=float, default=100.0, help="Distance from FOV center to plate y=0 (mm). Paper eval uses 3cm..19cm; MC example ~10cm.")
    # Use YOUR observed y-centers by default, because they are known-good with this repo.
    ap.add_argument("--y_l1", type=float, default=10.0)
    ap.add_argument("--y_l2", type=float, default=18.0)
    ap.add_argument("--y_l3", type=float, default=26.0)
    ap.add_argument("--y_l4", type=float, default=34.0)
    args = ap.parse_args()

    plate_thick = float(args.plate_thick)
    plate_y0, plate_y1 = 0.0, plate_thick

    # -----------------------------
    # Param_Image.dat
    # Convention used by README: FOV center is at y = shiftY - FOV2Collimator0
    # -----------------------------
    img = np.array([
        float(args.fov_nx), float(args.fov_ny), float(args.fov_nz),
        float(args.vox), float(args.vox), float(args.vox),
        1.0, 2.0*math.pi,
        0.0, 0.0, 0.0,
        float(args.fov2coll0)
    ], dtype=np.float32)

    # -----------------------------
    # Param_Physics.dat
    # -----------------------------
    phy = np.array([
        0.0,  # flagUsingCompton
        1.0,  # flagSavingPESysmat
        0.0,  # flagSavingComptonSysmat
        0.0,  # flagSavingPEComptonSysmat
        1.0,  # flagUsingSameEnergyWindow
        float(E_LOW), float(E_HIGH),
        float(E_TARGET),
        0.0, 0.0
    ], dtype=np.float32)

    # -----------------------------
    # Param_Collimator.dat (1 layer)
    # -----------------------------
    hole_r = HOLE_DIAM / 2.0
    plate_area = PLATE_W_X * PLATE_H_Z
    hole_area = math.pi * hole_r * hole_r
    num_holes = int(round((OPEN_FRAC * plate_area) / hole_area))

    holes = gen_holes_random_nonoverlap(num_holes, hole_r, seed=args.seed)

    # col format: [0]=numLayers, layer at (id+1)*10, holes at 100+hid*9
    ncol = 100 + len(holes)*9 + 32
    col = np.zeros(ncol, dtype=np.float32)
    col[0] = 1.0
    base = 10
    col[base+0] = float(len(holes))
    col[base+1] = PLATE_W_X
    col[base+2] = plate_thick
    col[base+3] = PLATE_H_Z
    col[base+4] = 0.0

    # Attenuation placeholders (you can plug tungsten mu later)
    col[base+5] = 1.0
    col[base+6] = 0.8
    col[base+7] = 0.2

    for hid,(x,z) in enumerate(holes):
        off = 100 + hid*9
        col[off+0] = float(x)
        col[off+1] = plate_y0
        col[off+2] = plate_y1
        col[off+3] = float(z)
        col[off+4] = hole_r
        col[off+5] = 0.0
        col[off+6] = 0.0
        col[off+7] = 0.0
        col[off+8] = 1.0

    # -----------------------------
    # Param_Detector.dat
    # Format: [0]=numDet, then numDet blocks of 12:
    # x,y,z,widthX,thickY,heightZ,mu_tot,mu_pe,mu_c,energyRes,rotY,flag
    # -----------------------------
    det_boxes = []

    # IMPORTANT FIX:
    # You previously got insane Z spans (like +-189mm) because you were pitching Z by 6mm.
    # For a mosaic grid, pitch should follow the in-plane pixel size (x dimension),
    # not the crystal depth. So Z pitch is 3mm for L1-3 and 2mm for L4.
    # We implement that by using pitch_z = sx (not sz) inside add_mosaic().
    add_mosaic(det_boxes, L123_NX, L123_NZ, L123_SIZE, float(args.y_l1))
    add_mosaic(det_boxes, L123_NX, L123_NZ, L123_SIZE, float(args.y_l2))
    add_mosaic(det_boxes, L123_NX, L123_NZ, L123_SIZE, float(args.y_l3))
    add_mosaic(det_boxes, L4_NX,   L4_NZ,   L4_SIZE,   float(args.y_l4))

    det_boxes = np.array(det_boxes, dtype=np.float32)
    numDet = det_boxes.shape[0]

    det = np.zeros(1 + 12*numDet, dtype=np.float32)
    det[0] = float(numDet)
    for i in range(numDet):
        x,y,z,sx,sy,sz = det_boxes[i].tolist()
        base = 1 + i*12
        det[base+0] = x
        det[base+1] = y
        det[base+2] = z
        det[base+3] = sx
        det[base+4] = sy
        det[base+5] = sz
        det[base+6] = 1.0
        det[base+7] = 0.8
        det[base+8] = 0.2
        det[base+9] = float(ENERGY_RES)
        det[base+10] = 0.0
        det[base+11] = 1.0

    # Optional padding to 80000 floats (your repo sometimes reads fixed arrays safely)
    if args.pad_fixed_80000:
        col = pad_to_len(col, 80000)
        det = pad_to_len(det, 80000)
        img = pad_to_len(img, 100)   # many people pad these too
        phy = pad_to_len(phy, 100)

    # Write files
    pref = args.out_prefix
    write_f32(f"{pref}_Image.dat", img)
    write_f32(f"{pref}_Physics.dat", phy)
    write_f32(f"{pref}_Collimator.dat", col)
    write_f32(f"{pref}_Detector.dat", det)

    # Print summary
    def span(vals): return float(vals.max() - vals.min())
    xyz = det.reshape(-1)[1:1+numDet*12].reshape(numDet,12)[:,0:3]
    print("WROTE:")
    print(f"  {pref}_Collimator.dat  holes={len(holes)}  plate_thick={plate_thick}mm")
    print(f"  {pref}_Detector.dat    numDet={numDet}")
    print(f"  {pref}_Image.dat       FOV=({args.fov_nx},{args.fov_ny},{args.fov_nz}) vox, vox={args.vox}mm, FOV2Coll0={args.fov2coll0}mm")
    print("Detector center spans:")
    print(f"  X span={span(xyz[:,0]):.3f}  Y span={span(xyz[:,1]):.3f}  Z span={span(xyz[:,2]):.3f}")
    print("Y planes:", sorted(np.unique(np.round(xyz[:,1],3)).tolist()))
    print("NOTE: Z pitch uses sx (3mm/2mm) so Z span will be sane (NOT 6mm pitch).")

if __name__ == "__main__":
    main()
