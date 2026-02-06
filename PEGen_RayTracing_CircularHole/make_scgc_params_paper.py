#!/usr/bin/env python3
import numpy as np
import math

# ============================================================
# Paper-aligned SCGC geometry (as described in the paper)
# ============================================================
# System envelope: 200 mm × 150 mm × 150 mm
# Coded-aperture: 12.5% open ratio, 1.6 mm diameter circular holes, random distribution
# Detectors:
#   L1-L3: 32×16, each crystal 3×3×6 mm (paper says x,y,z)
#   L4   : 64×64, each crystal 2×2×6 mm
#
# IMPORTANT: Repo conventions (README)
# - Param_Collimator[0] = numLayers
# - layer params are at [layer_id*10 + 0..7] BUT many codes assume layer0 starts at 10.
#   README says layer uses id*10; implementation often uses layer_id=1 offset.
#   To be safe with THIS repo (matches your successful run), we store layer0 at index 10.
# - Holes start at index 100, stride 9
# - Param_Detector[0] = numDetectorBins, then stride 12 starting at index 1
# - Param_Image has 12 floats
# - Param_Physics has 10 floats
#
# Coordinate choice (works with this repo in practice):
# - Put collimator plate spanning y in [0, PLATE_THICK]
# - Put detectors at positive y behind the plate
# - Put FOV in front of plate: FOV2Collimator0 > 0 (repo prints it directly)
# ============================================================

# ---------- Physics (paper) ----------
E_LOW, E_HIGH = 112.0, 168.0
E_TARGET = 140.0
ENERGY_RES = 0.20  # 20% @ 140 keV

# ---------- Plate (paper) ----------
PLATE_W_X = 200.0
PLATE_H_Z = 150.0
HOLE_DIAM = 1.6
OPEN_FRAC = 0.125

# Plate thickness is NOT explicitly in your pasted excerpt.
# Paper dataset spans 1..10 mm. If you don’t know, keep 5 mm (reasonable mid).
PLATE_THICK = 5.0

# ---------- Detector crystals (paper) ----------
L123_NX, L123_NZ = 32, 16
L123_SIZE = (3.0, 3.0, 6.0)  # x,y,z (mm)
L4_NX, L4_NZ = 64, 64
L4_SIZE = (2.0, 2.0, 6.0)

# Pitch: paper calls it "mosaic structure" / "densely arranged"
# Most likely pitch == crystal size in X and Z (no gaps).
L123_PITCH_X, L123_PITCH_Z = 3.0, 6.0
L4_PITCH_X,   L4_PITCH_Z   = 2.0, 6.0

# ---------- FOV ----------
# Repo GPUPTS eval uses 2D plane: 160×160 with 1mm pixels.
# Paper TL uses 3D: 160×160×160, but this repo’s PE sysmat path is typically 2D.
FOV_NX, FOV_NY, FOV_NZ = 160, 160, 1
VOX = (1.0, 1.0, 1.0)

# Distance from FOV plane to camera surface spans 30..230 mm in paper.
# Here, this field is "FOV2Collimator0(mm)" in README.
FOV2COLL0 = 30.0  # mm (start with 30 mm)

# Detector placement gaps (paper does not give exact internals in the excerpt)
# Keep small + configurable-like defaults; change if you later locate Fig/table with true gaps.
GAP_PLATE_TO_L1 = 2.0
GAP_L1_TO_L2    = 2.0
GAP_L2_TO_L3    = 2.0
GAP_L3_TO_L4    = 2.0

# ---------- Attenuation coefficients ----------
# If you don’t have material coefficients, keep placeholders.
# They affect physics but should not crash. (Crashes are almost always buffer/shape mismatches.)
MU_COL_TOT, MU_COL_PE, MU_COL_C = 1.0, 0.8, 0.2
MU_DET_TOT, MU_DET_PE, MU_DET_C = 1.0, 0.8, 0.2

# ============================================================
# Helpers
# ============================================================
def write_f32(path, arr):
    arr = np.asarray(arr, dtype=np.float32)
    with open(path, "wb") as f:
        f.write(arr.tobytes())

def gen_random_holes(seed=0):
    import random
    random.seed(seed)

    hole_r = HOLE_DIAM / 2.0
    plate_area = PLATE_W_X * PLATE_H_Z
    hole_area = math.pi * hole_r * hole_r
    num_holes_target = int(round((OPEN_FRAC * plate_area) / hole_area))

    holes = []
    max_tries = 2_000_000
    min_sep = HOLE_DIAM * 1.05  # mild non-overlap

    tries = 0
    while len(holes) < num_holes_target and tries < max_tries:
        tries += 1
        x = random.uniform(-PLATE_W_X/2 + hole_r, PLATE_W_X/2 - hole_r)
        z = random.uniform(-PLATE_H_Z/2 + hole_r, PLATE_H_Z/2 - hole_r)

        ok = True
        # cheap local check (last few only) to avoid O(N^2) blowup
        for (x2, z2) in holes[-250:]:
            if (x-x2)**2 + (z-z2)**2 < (min_sep**2):
                ok = False
                break
        if ok:
            holes.append((x, z))

    if len(holes) < num_holes_target:
        print(f"[WARN] placed {len(holes)}/{num_holes_target} holes (increase max_tries if you care).")

    return holes

def add_detector_layer(det_entries, nx, nz, size_xyz, pitch_x, pitch_z, y_center):
    sx, sy, sz = size_xyz
    for iz in range(nz):
        for ix in range(nx):
            x = (ix - (nx - 1)/2) * pitch_x
            z = (iz - (nz - 1)/2) * pitch_z
            det_entries.append([
                x, y_center, z,
                sx, sy, sz,
                MU_DET_TOT, MU_DET_PE, MU_DET_C,
                ENERGY_RES,
                0.0,   # rotation angle
                1.0    # flag
            ])

# ============================================================
# Build Param_Image.dat (12 floats)
# ============================================================
img = np.array([
    FOV_NX, FOV_NY, FOV_NZ,
    VOX[0], VOX[1], VOX[2],
    1.0, 0.0,     # numRotation, anglePerRotation (keep 0 if only 1 rot)
    0.0, 0.0, 0.0,
    float(FOV2COLL0),
], dtype=np.float32)
write_f32("Param_Image.dat", img)

# ============================================================
# Build Param_Physics.dat (10 floats)
# ============================================================
phy = np.array([
    0.0,   # flagUsingCompton (PE module)
    1.0,   # save PE
    0.0,   # save Compton
    0.0,   # save sum
    1.0,   # same energy window
    E_LOW, E_HIGH,
    E_TARGET,
    0.0, 0.0
], dtype=np.float32)
write_f32("Param_Physics.dat", phy)

# ============================================================
# Build Param_Collimator.dat
# ============================================================
holes = gen_random_holes(seed=0)
num_holes = len(holes)

# Storage length: must be large enough; repo reads 80000 floats anyway.
# But we still write a compact file (safe).
ncol = 100 + num_holes * 9 + 64
col = np.zeros(ncol, dtype=np.float32)

col[0] = 1.0  # num layers

# THIS repo behaves correctly when layer0 starts at index 10
layer0 = 10
col[layer0 + 0] = float(num_holes)
col[layer0 + 1] = float(PLATE_W_X)
col[layer0 + 2] = float(PLATE_THICK)
col[layer0 + 3] = float(PLATE_H_Z)
col[layer0 + 4] = 0.0
col[layer0 + 5] = float(MU_COL_TOT)
col[layer0 + 6] = float(MU_COL_PE)
col[layer0 + 7] = float(MU_COL_C)

# Plate spans y in [0, PLATE_THICK]
y1 = 0.0
y2 = float(PLATE_THICK)
hole_r = HOLE_DIAM / 2.0

for hid, (x, z) in enumerate(holes):
    off = 100 + hid * 9
    col[off + 0] = float(x)
    col[off + 1] = y1
    col[off + 2] = y2
    col[off + 3] = float(z)
    col[off + 4] = float(hole_r)
    col[off + 5] = 0.0
    col[off + 6] = 0.0
    col[off + 7] = 0.0
    col[off + 8] = 1.0

write_f32("Param_Collimator.dat", col)

# ============================================================
# Build Param_Detector.dat
# ============================================================
det_entries = []

# Place detector layers behind plate (+y)
# y_center = plate back face + gap + half thickness (y size)
plate_back = PLATE_THICK

y_l1 = plate_back + GAP_PLATE_TO_L1 + L123_SIZE[1]/2
y_l2 = y_l1 + L123_SIZE[1]/2 + GAP_L1_TO_L2 + L123_SIZE[1]/2
y_l3 = y_l2 + L123_SIZE[1]/2 + GAP_L2_TO_L3 + L123_SIZE[1]/2
y_l4 = y_l3 + L123_SIZE[1]/2 + GAP_L3_TO_L4 + L4_SIZE[1]/2

add_detector_layer(det_entries, L123_NX, L123_NZ, L123_SIZE, L123_PITCH_X, L123_PITCH_Z, y_l1)
add_detector_layer(det_entries, L123_NX, L123_NZ, L123_SIZE, L123_PITCH_X, L123_PITCH_Z, y_l2)
add_detector_layer(det_entries, L123_NX, L123_NZ, L123_SIZE, L123_PITCH_X, L123_PITCH_Z, y_l3)
add_detector_layer(det_entries, L4_NX,   L4_NZ,   L4_SIZE,   L4_PITCH_X,   L4_PITCH_Z,   y_l4)

num_det = len(det_entries)
det = np.zeros(1 + 12 * num_det, dtype=np.float32)
det[0] = float(num_det)

for i, row in enumerate(det_entries):
    base = 1 + i * 12
    det[base:base+12] = np.asarray(row, dtype=np.float32)

write_f32("Param_Detector.dat", det)

print("Wrote Param_*.dat")
print(f"  holes: {num_holes}")
print(f"  detectors: {num_det}")
print(f"  FOV: {FOV_NX}×{FOV_NY}×{FOV_NZ}")
print(f"  FOV2Collimator0: {FOV2COLL0} mm")
