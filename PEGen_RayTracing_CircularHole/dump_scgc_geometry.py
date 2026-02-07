#!/usr/bin/env python3
import argparse
import numpy as np
import os

def f32(path):
    return np.fromfile(path, dtype=np.float32)

def bbox_of_centers(pts):
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    return mn, mx, (mx - mn)

def uniq_rounded(a, decimals=3):
    ar = np.round(a, decimals=decimals)
    # unique rows
    ar2 = ar.view([('', ar.dtype)] * ar.shape[1])
    u = np.unique(ar2).view(ar.dtype).reshape(-1, ar.shape[1])
    return u

def parse_collimator(col):
    num_layers = int(round(col[0]))
    layers = []
    holes_all = []
    for lid in range(num_layers):
        base = (lid + 1) * 10
        num_holes = int(round(col[base + 0]))
        wX = float(col[base + 1])
        thickY = float(col[base + 2])
        hZ = float(col[base + 3])
        dist = float(col[base + 4])
        mu_tot = float(col[base + 5])
        mu_pe  = float(col[base + 6])
        mu_c   = float(col[base + 7])
        layers.append((num_holes, wX, thickY, hZ, dist, mu_tot, mu_pe, mu_c))

        # holes live at 100 + hid*9
        holes = []
        for hid in range(num_holes):
            off = 100 + hid * 9
            x  = float(col[off + 0])
            y1 = float(col[off + 1])
            y2 = float(col[off + 2])
            z  = float(col[off + 3])
            r  = float(col[off + 4])
            holes.append((x, y1, y2, z, r))
        holes_all.append(np.array(holes, dtype=np.float32))
    return num_layers, layers, holes_all

def parse_detector(det):
    num_det = int(round(det[0]))
    body = det[1:]
    if body.size < num_det * 12:
        raise RuntimeError(f"Detector file too short: body floats={body.size}, expected={num_det*12}")
    body = body[:num_det*12].reshape(num_det, 12)
    # fields
    xyz = body[:, 0:3]
    size = body[:, 3:6]   # widthX, thickY, heightZ
    mu = body[:, 6:9]
    eres = body[:, 9]
    rot = body[:, 10]
    flag = body[:, 11]
    return num_det, body, xyz, size, mu, eres, rot, flag

def parse_image(img):
    nx, ny, nz = map(int, np.round(img[0:3]))
    sx, sy, sz = map(float, img[3:6])
    nrot = int(round(img[6]))
    ang = float(img[7])
    shift = np.array(img[8:11], dtype=np.float32)
    fov2col0 = float(img[11])
    return nx, ny, nz, sx, sy, sz, nrot, ang, shift, fov2col0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--param_collimator", required=True)
    ap.add_argument("--param_detector", required=True)
    ap.add_argument("--param_image", required=True)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    print("=== FILES ===")
    for p in [args.param_collimator, args.param_detector, args.param_image]:
        print(f"{p}: bytes={os.path.getsize(p)} floats={os.path.getsize(p)//4}")
    if args.label:
        print(f"Label: {args.label}")
    print()

    img = f32(args.param_image)
    col = f32(args.param_collimator)
    det = f32(args.param_detector)

    print("=== Param_Image ===")
    nx, ny, nz, sx, sy, sz, nrot, ang, shift, fov2col0 = parse_image(img)
    fov_size = np.array([nx*sx, ny*sy, nz*sz], dtype=np.float32)
    fov_center = shift + np.array([0.0, -fov2col0, 0.0], dtype=np.float32)  # repo convention
    fov_min = fov_center - 0.5 * fov_size
    fov_max = fov_center + 0.5 * fov_size
    print(f"voxels (nx,ny,nz)=({nx},{ny},{nz})  voxel_size=({sx},{sy},{sz}) mm")
    print(f"FOV size (mm) = {fov_size.tolist()}")
    print(f"shift = {shift.tolist()} mm")
    print(f"FOV2Collimator0 = {fov2col0} mm")
    print(f"FOV center (x,y,z) = {fov_center.tolist()}")
    print(f"FOV bbox min = {fov_min.tolist()}")
    print(f"FOV bbox max = {fov_max.tolist()}")
    print(f"numRotation={nrot}  anglePerRotation={ang}")
    print()

    print("=== Param_Collimator ===")
    num_layers, layers, holes_all = parse_collimator(col)
    print(f"numLayers = {num_layers}")
    # Only paper uses 1 plate layer; still print generically
    plate_y0 = None
    plate_y1 = None
    for lid, (num_holes, wX, thickY, hZ, dist, mu_tot, mu_pe, mu_c) in enumerate(layers):
        print(f"Layer{lid}: numHoles={num_holes} widthX={wX}mm thickY={thickY}mm heightZ={hZ}mm dist={dist}mm mu(tot,pe,c)=({mu_tot},{mu_pe},{mu_c})")
        holes = holes_all[lid]
        print(f"  holes parsed = {holes.shape[0]}")
        rs = holes[:,4]
        print(f"  hole radius stats: min={rs.min():.4f} max={rs.max():.4f} mean={rs.mean():.4f}")
        y1u = (holes[:,1].min(), holes[:,1].max())
        y2u = (holes[:,2].min(), holes[:,2].max())
        print(f"  hole y1 range=({y1u[0]:.3f},{y1u[1]:.3f})  y2 range=({y2u[0]:.3f},{y2u[1]:.3f})")
        print(f"  hole center bbox X=[{holes[:,0].min():.3f},{holes[:,0].max():.3f}]  Z=[{holes[:,3].min():.3f},{holes[:,3].max():.3f}]")
        if plate_y0 is None:
            plate_y0 = float(holes[0,1])
            plate_y1 = float(holes[0,2])
        # print sample holes
        print("  sample holes (x,y1,y2,z,r):")
        for i in range(min(5, holes.shape[0])):
            x,y1,y2,z,r = holes[i]
            print(f"    {i}: ({x:.3f},{y1:.3f},{y2:.3f},{z:.3f},{r:.3f})")
    print()

    print("=== Param_Detector ===")
    num_det, body, xyz, size, mu, eres, rot, flag = parse_detector(det)
    print(f"numDet = {num_det}")
    mn, mx, sp = bbox_of_centers(xyz)
    print(f"det center bbox min = {mn.tolist()}")
    print(f"det center bbox max = {mx.tolist()}")
    print(f"det center span     = {sp.tolist()}")
    usz = uniq_rounded(size, 3)
    print(f"unique sizes (widthX,thickY,heightZ) count={usz.shape[0]}:")
    for row in usz:
        print(f"  {row.tolist()}")
    print(f"y stats: min={xyz[:,1].min():.3f} max={xyz[:,1].max():.3f} mean={xyz[:,1].mean():.3f}")

    # group by Y plane (layering)
    y_round = np.round(xyz[:,1], 3)
    ys, counts = np.unique(y_round, return_counts=True)
    order = np.argsort(ys)
    ys, counts = ys[order], counts[order]
    print("Y planes (rounded 0.001):")
    for y,c in zip(ys, counts):
        print(f"  y={y:.3f}  count={int(c)}")

    # Approx paper expectations (center spans)
    print()
    print("=== Paper expectation checkpoints (centers) ===")
    print("Plate: X≈200mm, Z≈150mm, holes diameter=1.6mm, open frac=12.5%")
    print("L1-3: 32x16 of 3x3x6 => center span X≈(32-1)*3=93mm, Z≈(16-1)*3=45mm")
    print("L4  : 64x64 of 2x2x6 => center span X≈(64-1)*2=126mm, Z≈(64-1)*2=126mm")
    print("FOV: 2D eval plane 160x160 at distance (paper eval often 3cm..19cm, MC at 10cm)")
    print()

    # sanity distances
    print("=== Relative positioning sanity ===")
    print(f"Assumed plate y-range from holes: y in [{plate_y0:.3f},{plate_y1:.3f}]")
    print(f"Nearest detector center y = {xyz[:,1].min():.3f} (should be > plate_y1)")
    print(f"FOV center y = {fov_center[1]:.3f} (should be < plate_y0)")
    print(f"FOV max y = {fov_max[1]:.3f} (should be < plate_y0 ideally for clean 2D plane)")
    if fov_max[1] >= plate_y0:
        print("WARNING: FOV plane extends to y>=0 (overlaps/behind plate). For 2D eval you typically want it fully in front.")
    if xyz[:,1].min() <= plate_y1:
        print("WARNING: some detector centers are inside/at plate. Wrong.")
    if xyz[:,1].min() < 0:
        print("WARNING: detectors in front of plate (y<0). Wrong for this repo convention.")
    print()

if __name__ == "__main__":
    main()
