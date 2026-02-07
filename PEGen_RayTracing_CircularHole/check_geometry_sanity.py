#!/usr/bin/env python3
import argparse
import numpy as np

def read_f32(path):
    return np.fromfile(path, dtype=np.float32)

def bbox_points(xyz):
    mn = xyz.min(axis=0)
    mx = xyz.max(axis=0)
    return mn, mx, mx - mn

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--param_collimator", required=True)
    ap.add_argument("--param_detector", required=True)
    ap.add_argument("--param_image", required=True)
    ap.add_argument("--max_holes_print", type=int, default=5)
    args = ap.parse_args()

    col = read_f32(args.param_collimator)
    det = read_f32(args.param_detector)
    img = read_f32(args.param_image)

    # -------------------------
    # Image/FOV
    # -------------------------
    nx, ny, nz = map(int, img[0:3])
    vx, vy, vz = img[3:6]
    num_rot = int(img[6])
    ang_per = img[7]
    shiftx, shifty, shiftz = img[8:11]
    fov2c0 = img[11]

    fov_size = np.array([nx*vx, ny*vy, nz*vz], dtype=float)
    fov_center = np.array([shiftx, -fov2c0 + shifty, shiftz], dtype=float)  # NOTE: y is in front of plate
    fov_min = fov_center - 0.5*fov_size
    fov_max = fov_center + 0.5*fov_size

    print("=== Param_Image ===")
    print(f"voxels (nx,ny,nz)=({nx},{ny},{nz})  voxel_size=({vx},{vy},{vz}) mm")
    print(f"FOV size (mm) = {fov_size.tolist()}")
    print(f"shift = ({shiftx},{shifty},{shiftz}) mm")
    print(f"FOV2Collimator0 = {fov2c0} mm")
    print(f"FOV center (x,y,z) = {fov_center.tolist()}")
    print(f"FOV bbox min = {fov_min.tolist()}")
    print(f"FOV bbox max = {fov_max.tolist()}")
    print(f"numRotation={num_rot}  anglePerRotation={ang_per}")

    # -------------------------
    # Collimator (plate + holes)
    # -------------------------
    num_layers = int(round(col[0]))
    print("\n=== Param_Collimator ===")
    print(f"numLayers = {num_layers}")

    # layer 0 layout: at index 10..17 per README
    # [layer*10+0]: numHoles
    # [layer*10+1]: widthX, [layer*10+2]: thickY, [layer*10+3]: heightZ
    # [layer*10+4]: dist to 1st layer (layer 0 should be 0)
    if num_layers < 1:
        raise RuntimeError("No collimator layers found.")

    l0 = 10
    num_holes = int(round(col[l0+0]))
    plate_wx = float(col[l0+1])
    plate_th = float(col[l0+2])
    plate_hz = float(col[l0+3])
    plate_dist = float(col[l0+4])

    print(f"Layer0: numHoles={num_holes}  widthX={plate_wx}mm  thickY={plate_th}mm  heightZ={plate_hz}mm  dist={plate_dist}mm")

    # holes: 9 floats each starting at 100
    holes = []
    for hid in range(num_holes):
        off = 100 + hid*9
        if off + 8 >= len(col):
            print(f"WARNING: col file too short for hole {hid} (off={off}).")
            break
        x = float(col[off+0])
        y1 = float(col[off+1])
        y2 = float(col[off+2])
        z = float(col[off+3])
        r = float(col[off+4])
        holes.append((x,y1,y2,z,r))

    holes = np.array(holes, dtype=float)
    if holes.size > 0:
        # hole centers in x,z ; y uses y1/y2
        xz = holes[:, [0,3]]
        print(f"Holes parsed = {holes.shape[0]}")
        print(f"Hole radius stats (mm): min={holes[:,4].min():.4f} max={holes[:,4].max():.4f} mean={holes[:,4].mean():.4f}")
        print(f"Hole y1/y2 unique-ish: y1(min,max)=({holes[:,1].min():.3f},{holes[:,1].max():.3f})  y2(min,max)=({holes[:,2].min():.3f},{holes[:,2].max():.3f})")

        x_min, z_min = xz.min(axis=0)
        x_max, z_max = xz.max(axis=0)
        print(f"Hole center bbox X: [{x_min:.3f}, {x_max:.3f}]  Z: [{z_min:.3f}, {z_max:.3f}]")

        print("\nSample holes (x,y1,y2,z,r):")
        for i in range(min(args.max_holes_print, holes.shape[0])):
            print(f"  {i}: {holes[i].tolist()}")

    # -------------------------
    # Detectors
    # -------------------------
    num_det = int(round(det[0]))
    if 1 + 12*num_det > len(det):
        raise RuntimeError(f"Detector file length mismatch: det_len={len(det)} floats but expects >= {1+12*num_det}")

    D = det[1:].reshape(num_det, 12)
    xyz = D[:, 0:3]
    size = D[:, 3:6]   # widthX, thickY, heightZ
    rot = D[:, 10]
    flag = D[:, 11]

    print("\n=== Param_Detector ===")
    print(f"numDet = {num_det}")
    mn, mx, span = bbox_points(xyz)
    print(f"Detector center bbox min = {mn.tolist()}")
    print(f"Detector center bbox max = {mx.tolist()}")
    print(f"Detector center span     = {span.tolist()}")

    # Show unique sizes
    sizes_unique = np.unique(np.round(size, 6), axis=0)
    print(f"Unique detector sizes (widthX,thickY,heightZ) count={len(sizes_unique)}:")
    for s in sizes_unique:
        print(f"  {s.tolist()}")

    # Check Y ordering
    y = xyz[:,1]
    print(f"Detector Y stats: min={y.min():.3f} max={y.max():.3f} mean={y.mean():.3f}")

    # Rough clustering of layers by Y coordinate
    # bucket by rounding to 0.01 mm
    y_round = np.round(y, 2)
    uniq_y, counts = np.unique(y_round, return_counts=True)
    top = np.argsort(counts)[::-1][:10]
    print("\nTop Y-planes (rounded to 0.01mm):")
    for idx in top:
        print(f"  y={uniq_y[idx]:.2f}  count={counts[idx]}")

    # Check if any detectors overlap into plate zone or FOV zone
    # Plate assumed at y in [0, plate_th]
    in_plate = np.logical_and(y >= 0.0 - 1e-6, y <= plate_th + 1e-6).sum()
    in_front = (y < 0.0).sum()
    print(f"\nDetectors with center y in plate thickness [0, {plate_th}]: {in_plate}")
    print(f"Detectors with center y < 0 (in front of plate): {in_front}")

    # Check FOV relative to plate
    print("\n=== FOV vs Plate sanity ===")
    print(f"Plate assumed at y in [0, {plate_th}]")
    print(f"FOV bbox y-range = [{fov_min[1]:.3f}, {fov_max[1]:.3f}]")
    if fov_max[1] > 0:
        print("WARNING: FOV extends into +Y (behind/into plate region). That is usually wrong.")
    if fov_max[1] > 0 and fov_min[1] < 0:
        print("WARNING: FOV straddles y=0 (overlaps plate). Likely wrong.")
    if fov_max[1] <= 0:
        print("OK: FOV entirely in front of plate (y<=0).")

    # Paper expectation hints
    print("\n=== Paper expectation checkpoints (for you) ===")
    print("Expected plate: 200 x 150 mm in X/Z; thickness 1..10mm (paper dataset range).")
    print("Expected L1-3 mosaic approx: X span ~ 32*3=96mm; Z span ~16*3=48mm (centers).")
    print("Expected L4 mosaic approx: X span ~64*2=128mm; Z span ~64*2=128mm (centers).")
    print("Expected FOV cube: 160x160x160mm centered at y=-distance (distance 30..230mm).")

if __name__ == "__main__":
    main()
