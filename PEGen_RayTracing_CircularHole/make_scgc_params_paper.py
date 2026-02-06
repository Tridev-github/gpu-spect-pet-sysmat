#!/usr/bin/env python3
import numpy as np
import math

# ============================================================
# Paper constants (from your pasted text)
# ============================================================
E_LOW, E_HIGH = 112.0, 168.0
E_TARGET = 140.0
ENERGY_RES = 0.20  # 20% at 140 keV

# System outer size is 200 x 150 x 150 mm (paper). Only XY face explicitly used for plate.
PLATE_W_X = 200.0  # mm
PLATE_H_Z = 150.0  # mm
HOLE_DIAM = 1.6    # mm
OPEN_FRAC = 0.125  # 12.5%

# Plate thickness is NOT explicitly stated in your excerpt.
# The dataset varies 1..10 mm; many people use ~5 mm as nominal.
PLATE_THICK = 5.0  # mm (override if you later confirm from Fig.1 / table)

# Detector mosaic counts (paper)
L123_NX, L123_NZ = 32, 16
L4_NX,   L4_NZ   = 64, 64

# Paper detector sizes (interpretation for this repo):
# Paper gives 3(x) x 3(y) x 6(z). In imaging hardware, the 6mm is thickness/depth.
# Repo expects: width=X, thickness=Y, height=Z.
L123_WX, L123_TY, L123_HZ = 3.0, 6.0, 3.0
L4_WX,   L4_TY,   L4_HZ   = 2.0, 6.0, 2.0

# Assume mosaic pitch equals in-plane size (no gap). If you later find gap/pitch in paper, change.
L123_PITCH_X, L123_PITCH_Z = 3.0, 3.0
L4_PITCH_X,   L4_PITCH_Z   = 2.0, 2.0

# FOV (paper: 16cm cube, 1mm voxel)
FOV_NX, FOV_NY, FOV_NZ = 160, 160, 160
VOX_X, VOX_Y, VOX_Z = 1.0, 1.0, 1.0

# Distance to camera surface in paper spans 30..230mm.
# Repo Param_Image[11] prints "FOV center to 1st Collimator".
# Use 30 mm nominal (matches paper mention and your run output).
FOV2COLL0 = 30.0

# Layer spacings are NOT explicitly stated in your excerpt.
# Keep tight but configurable defaults (mm). If you later get exact distances from Fig 1(b), update these.
GAP_PLATE_TO_L1 = 2.0
GAP_L1_TO_L2    = 2.0
GAP_L2_TO_L3    = 2.0
GAP_L3_TO_L4    = 2.0

# ============================================================
# Helpers
# ============================================================
def write_f32(path, arr):
    arr = np.asarray(arr, dtype=np.float32)
    with open(path, "wb") as f:
        f.write(arr.tobytes())

def poissonish_holes(width, height, radius, target_n, seed=0, min_sep=None, edge_margin=None):
    """
    Fast rejection w/ grid accel.
    Returns (N,2) array of (x,z).
    """
    rng = np.random.default_rng(seed)
    if min_sep is None:
        min_sep = 2.0 * radius * 1.05
    if edge_margin is None:
        edge_margin = radius

    x_min = -width/2 + edge_margin
    x_max =  width/2 - edge_margin
    z_min = -height/2 + edge_margin
    z_max =  height/2 - edge_margin

    cell = min_sep / math.sqrt(2.0)
    grid = {}
    pts = []
    min_sep2 = min_sep * min_sep
    max_attempts = 5_000_000
    attempts = 0

    def cell_key(x,z):
        return (int(math.floor(x / cell)), int(math.floor(z / cell)))

    while len(pts) < target_n and attempts < max_attempts:
        attempts += 1
        x = rng.uniform(x_min, x_max)
        z = rng.uniform(z_min, z_max)
        ck = cell_key(x,z)

        ok = True
        for dx in (-1,0,1):
            for dz in (-1,0,1):
                nk = (ck[0]+dx, ck[1]+dz)
                if nk not in grid:
                    continue
                for (px,pz) in grid[nk]:
                    if (x-px)*(x-px) + (z-pz)*(z-pz) < min_sep2:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break

        if ok:
            pts.append((x,z))
            grid.setdefault(ck, []).append((x,z))

    return np.array(pts, dtype=np.float32)

# ============================================================
# Build Param_Image.dat (12 floats)
# ============================================================
param_image = np.array([
    float(FOV_NX), float(FOV_NY), float(FOV_NZ),
    float(VOX_X),  float(VOX_Y),  float(VOX_Z),
    1.0,           0.0,           # numRotation=1, anglePerRotation=0
    0.0, 0.0, 0.0,                 # shifts
    float(FOV2COLL0)
], dtype=np.float32)
write_f32("Param_Image.dat", param_image)

# ============================================================
# Build Param_Physics.dat (>=10 floats)
# ============================================================
param_physics = np.array([
    0.0,   # flagUsingCompton (PE module ignores it; keep 0 here)
    1.0,   # flagSavingPESysmat
    0.0,   # flagSavingComptonSysmat
    0.0,   # flagSavingPEComptonSysmat
    1.0,   # flagUsingSameEnergyWindow
    float(E_LOW), float(E_HIGH),
    float(E_TARGET),
    0.0,   # flagCalculateCrystalGeometryRelationship
    0.0    # flagCalculateCollimatorGeometryRelationship
], dtype=np.float32)
write_f32("Param_Physics.dat", param_physics)

# ============================================================
# Build Param_Collimator.dat
# Layout per README:
# col[0] = numLayers
# layer params at id*10 (id=0 => indices 0..9, but index0 is numLayers)
# holes at 100 + hid*9
#
# IMPORTANT: We will store layer0 parameters starting at index 10 (id=1*10),
# to avoid clobbering col[0]. This matches how most users succeed with this repo.
# ============================================================
hole_r = HOLE_DIAM / 2.0
plate_area = PLATE_W_X * PLATE_H_Z
hole_area = math.pi * hole_r * hole_r
target_holes = int(round(OPEN_FRAC * plate_area / hole_area))

holes = poissonish_holes(
    width=PLATE_W_X,
    height=PLATE_H_Z,
    radius=hole_r,
    target_n=target_holes,
    seed=0,
    min_sep=HOLE_DIAM * 1.05,
    edge_margin=hole_r
)
num_holes = int(holes.shape[0])

# allocate
ncol = 100 + num_holes*9 + 32
col = np.zeros(ncol, dtype=np.float32)

col[0] = 1.0  # one plate layer

layer0 = 10
col[layer0+0] = float(num_holes)
col[layer0+1] = float(PLATE_W_X)
col[layer0+2] = float(PLATE_THICK)
col[layer0+3] = float(PLATE_H_Z)
col[layer0+4] = 0.0

# NOTE: the repo expects attenuation coefficients; paper doesn't list numbers in excerpt.
# Use placeholders that keep math stable. Replace later if you have mu values.
col[layer0+5] = 1.0
col[layer0+6] = 0.8
col[layer0+7] = 0.2

# Plate is at y in [0, PLATE_THICK] and "Y of 1st collimator = 0" (repo note).
y1 = 0.0
y2 = PLATE_THICK

for hid, (x,z) in enumerate(holes):
    off = 100 + hid*9
    col[off+0] = float(x)
    col[off+1] = float(y1)
    col[off+2] = float(y2)
    col[off+3] = float(z)
    col[off+4] = float(hole_r)
    col[off+5] = 0.0
    col[off+6] = 0.0
    col[off+7] = 0.0
    col[off+8] = 1.0

write_f32("Param_Collimator.dat", col)

# ============================================================
# Build Param_Detector.dat
# det[0] = numDetectorBins
# per det: 12 floats at base=1+i*12:
# x,y,z, widthX, thicknessY, heightZ, muTot, muPE, muC, energyRes, rotAngle, flag
# ============================================================
det_entries = []

def add_layer(nx, nz, pitch_x, pitch_z, wx, ty, hz, y_center):
    x0 = -(nx - 1) * pitch_x / 2.0
    z0 = -(nz - 1) * pitch_z / 2.0
    for iz in range(nz):
        for ix in range(nx):
            x = x0 + ix * pitch_x
            z = z0 + iz * pitch_z
            det_entries.append([
                x, y_center, z,
                wx, ty, hz,
                1.0, 0.8, 0.2,   # placeholder mu values
                ENERGY_RES,
                0.0,
                1.0
            ])

# Place layers behind the plate (positive Y)
# Plate back face at y=PLATE_THICK.
y_l1 = PLATE_THICK + GAP_PLATE_TO_L1 + L123_TY/2.0
y_l2 = y_l1 + L123_TY/2.0 + GAP_L1_TO_L2 + L123_TY/2.0
y_l3 = y_l2 + L123_TY/2.0 + GAP_L2_TO_L3 + L123_TY/2.0
y_l4 = y_l3 + L123_TY/2.0 + GAP_L3_TO_L4 + L4_TY/2.0

add_layer(L123_NX, L123_NZ, L123_PITCH_X, L123_PITCH_Z, L123_WX, L123_TY, L123_HZ, y_l1)
add_layer(L123_NX, L123_NZ, L123_PITCH_X, L123_PITCH_Z, L123_WX, L123_TY, L123_HZ, y_l2)
add_layer(L123_NX, L123_NZ, L123_PITCH_X, L123_PITCH_Z, L123_WX, L123_TY, L123_HZ, y_l3)
add_layer(L4_NX,   L4_NZ,   L4_PITCH_X,   L4_PITCH_Z,   L4_WX,   L4_TY,   L4_HZ,   y_l4)

num_det = len(det_entries)
det = np.zeros(1 + 12*num_det, dtype=np.float32)
det[0] = float(num_det)
for i, row in enumerate(det_entries):
    base = 1 + i*12
    det[base:base+12] = np.asarray(row, dtype=np.float32)

write_f32("Param_Detector.dat", det)

print("WROTE Param_*.dat")
print("holes:", num_holes)
print("detBins:", num_det)
print("FOV:", f"{FOV_NX}x{FOV_NY}x{FOV_NZ} (voxels={FOV_NX*FOV_NY*FOV_NZ})")
print("NOTE: L123 mapped as width=3, thickness=6, height=3 to match paper depth=6mm.")
