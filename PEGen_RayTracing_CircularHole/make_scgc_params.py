import numpy as np
import math
import random

# -----------------------------
# Paper-driven constants
# -----------------------------
E_LOW, E_HIGH = 112.0, 168.0
E_TARGET = 140.0
ENERGY_RES = 0.20  # 20% @ 140keV

# Coded aperture plate (paper)
PLATE_W_X = 200.0  # mm (paper: system 200 mm x 150 mm x 150 mm; using 200x150 face)
PLATE_H_Z = 150.0  # mm
HOLE_DIAM = 1.6    # mm
OPEN_FRAC = 0.125  # 12.5% open ratio
PLATE_THICK = 5.0  # mm  <-- NOT in your pasted text; set from Fig/table when you find it.

# Detector crystals (paper)
L123_NX, L123_NZ = 32, 16
L123_SIZE = (3.0, 3.0, 6.0)  # (x,y,z) mm in paper wording
L4_NX, L4_NZ = 64, 64
L4_SIZE = (2.0, 2.0, 6.0)

# FOV (paper uses 16cm cube in TL section; MC eval uses 160x160mm plane)
FOV_NX, FOV_NY, FOV_NZ = 160, 160, 1
VOX = (1.0, 1.0, 1.0)

# Distances (THIS is where the paper excerpt is incomplete)
FOV2COLL0 = 30.0  # mm (paper says distance to surface ranges 30..230mm)
# Place plate centered near y=0..PLATE_THICK and detector layers at +Y.
GAP_PLATE_TO_L1 = 2.0   # mm  <-- set from paper Fig/table
GAP_L1_TO_L2    = 2.0   # mm  <-- set from paper Fig/table
GAP_L2_TO_L3    = 2.0   # mm
GAP_L3_TO_L4    = 2.0   # mm

# Coordinate convention:
# - Collimator plate spans y in [0, PLATE_THICK]
# - Detectors are placed at y > PLATE_THICK (IMPORTANT: +Y)
# - FOV center is at y = -FOV2COLL0 (so it's in front of plate along -Y)
# This matches your empirical finding: detector y must be +.
PLATE_Y0 = 0.0
PLATE_Y1 = PLATE_THICK
FOV_CENTER_Y = -FOV2COLL0

random.seed(0)

def write_f32(fname, arr):
    arr = np.asarray(arr, dtype=np.float32)
    open(fname, "wb").write(arr.tobytes())

# -----------------------------
# Params_Image.dat (len >= 12)
# -----------------------------
img = np.array([
    FOV_NX, FOV_NY, FOV_NZ,
    VOX[0], VOX[1], VOX[2],
    1, 2*math.pi,          # numRotation, anglePerRotation
    0.0, 0.0, 0.0,         # shifts
    FOV2COLL0
], dtype=np.float32)
write_f32("Params_Image.dat", img)

# -----------------------------
# Params_Physics.dat (len >= 10)
# -----------------------------
phy = np.array([
    0,      # flagUsingCompton (PE module ignores Compton anyway)
    1,      # flagSavingPESysmat
    0,      # flagSavingComptonSysmat
    0,      # flagSaving PE+Compton
    1,      # flagUsingSameEnergyWindow
    E_LOW, E_HIGH,
    E_TARGET,
    0, 0
], dtype=np.float32)
write_f32("Params_Physics.dat", phy)

# -----------------------------
# Params_Collimator.dat
# Layout (repo README):
# col[0] = numLayers
# layer params at (layer+1)*10
# holes at 100 + hole_id*9
# -----------------------------
hole_r = HOLE_DIAM / 2.0
plate_area = PLATE_W_X * PLATE_H_Z
hole_area = math.pi * hole_r * hole_r
num_holes = int(round((OPEN_FRAC * plate_area) / hole_area))

# Generate random non-overlapping-ish holes (simple rejection; not perfect, but works)
holes = []
max_tries = 2_000_000
tries = 0
min_sep = HOLE_DIAM * 1.05
while len(holes) < num_holes and tries < max_tries:
    tries += 1
    x = random.uniform(-PLATE_W_X/2 + hole_r, PLATE_W_X/2 - hole_r)
    z = random.uniform(-PLATE_H_Z/2 + hole_r, PLATE_H_Z/2 - hole_r)
    ok = True
    # crude spacing to avoid obvious overlaps
    for (x2, z2) in holes[-200:]:
        if (x-x2)**2 + (z-z2)**2 < (min_sep**2):
            ok = False
            break
    if ok:
        holes.append((x, z))

if len(holes) < num_holes:
    print(f"WARNING: only placed {len(holes)}/{num_holes} holes. Increase max_tries or relax min_sep.")
    num_holes = len(holes)

ncol = 100 + num_holes*9 + 32
col = np.zeros(ncol, dtype=np.float32)
col[0] = 1  # one plate layer

layer0 = 10
col[layer0+0] = num_holes
col[layer0+1] = PLATE_W_X
col[layer0+2] = PLATE_THICK
col[layer0+3] = PLATE_H_Z
col[layer0+4] = 0.0

# Attenuation coefficients:
# The repo expects mu_total/mu_PE/mu_Compton.
# You can plug real tungsten values later; placeholders below still let you run.
col[layer0+5] = 1.0
col[layer0+6] = 0.8
col[layer0+7] = 0.2

for hid, (x,z) in enumerate(holes):
    off = 100 + hid*9
    col[off+0] = x
    col[off+1] = PLATE_Y0
    col[off+2] = PLATE_Y1
    col[off+3] = z
    col[off+4] = hole_r
    col[off+5] = 0.0  # air hole
    col[off+6] = 0.0
    col[off+7] = 0.0
    col[off+8] = 1.0

write_f32("Params_Collimator.dat", col)

# -----------------------------
# Params_Detector.dat
# Format: det[0]=numDet; per detector: 12 floats at base=1+i*12:
# x,y,z,width,thickness,height,mu_total,mu_PE,mu_C,energyRes,rotAngle,flag
# NOTE: MUST place detectors at +Y for this code path to produce nonzeros (as you observed).
# -----------------------------
det_entries = []

def add_layer(nx, nz, size_xyz, y_center):
    sx, sy, sz = size_xyz
    pitch_x = sx
    pitch_z = sz if nz > 1 else sz
    # But crystals are "mosaic", usually pitch = size (no gaps) unless stated.
    pitch_x = sx
    pitch_z = sz if False else (sz if nz == 1 else sz)  # keep simple
    # Actually arrays are in X-Z plane; use pitch = size in X and Z:
    pitch_x = sx
    pitch_z = sz  # <-- if you know true pitch/gaps, set here

    for iz in range(nz):
        for ix in range(nx):
            x = (ix - (nx-1)/2) * pitch_x
            z = (iz - (nz-1)/2) * pitch_z
            det_entries.append([
                x, y_center, z,
                sx, sy, sz,
                1.0, 0.8, 0.2,
                ENERGY_RES,
                0.0,
                1.0
            ])

# Place layers behind the plate (positive y)
y_l1 = PLATE_THICK + GAP_PLATE_TO_L1 + L123_SIZE[1]/2
y_l2 = y_l1 + L123_SIZE[1]/2 + GAP_L1_TO_L2 + L123_SIZE[1]/2
y_l3 = y_l2 + L123_SIZE[1]/2 + GAP_L2_TO_L3 + L123_SIZE[1]/2
y_l4 = y_l3 + L123_SIZE[1]/2 + GAP_L3_TO_L4 + L4_SIZE[1]/2

add_layer(L123_NX, L123_NZ, L123_SIZE, y_l1)
add_layer(L123_NX, L123_NZ, L123_SIZE, y_l2)
add_layer(L123_NX, L123_NZ, L123_SIZE, y_l3)
add_layer(L4_NX,   L4_NZ,   L4_SIZE,   y_l4)

numDet = len(det_entries)
det = np.zeros(1 + 12*numDet, dtype=np.float32)
det[0] = numDet

for i, row in enumerate(det_entries):
    base = 1 + i*12
    det[base:base+12] = np.array(row, dtype=np.float32)

write_f32("Params_Detector.dat", det)

print("WROTE Params_*.dat")
print("num holes:", num_holes)
print("num detectors:", numDet)
print("FOV voxels:", FOV_NX*FOV_NY*FOV_NZ)
print("Output floats (det*vox*rot):", numDet*(FOV_NX*FOV_NY*FOV_NZ)*1)
