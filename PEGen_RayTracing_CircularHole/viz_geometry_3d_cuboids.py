#!/usr/bin/env python3
import numpy as np
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def read_f32(path):
    return np.fromfile(path, np.float32)

def cuboid_faces(cx, cy, cz, sx, sy, sz):
    # center + sizes -> 8 corners
    x0, x1 = cx - sx/2, cx + sx/2
    y0, y1 = cy - sy/2, cy + sy/2
    z0, z1 = cz - sz/2, cz + sz/2
    v = np.array([
        [x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],
        [x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1],
    ], dtype=float)
    # 6 faces (each as 4 verts)
    faces = [
        [v[0],v[1],v[2],v[3]],
        [v[4],v[5],v[6],v[7]],
        [v[0],v[1],v[5],v[4]],
        [v[2],v[3],v[7],v[6]],
        [v[1],v[2],v[6],v[5]],
        [v[0],v[3],v[7],v[4]],
    ]
    return faces

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--param_collimator", required=True)
    ap.add_argument("--param_detector", required=True)
    ap.add_argument("--param_image", required=True)
    ap.add_argument("--max_det", type=int, default=800, help="Plot at most this many detector cuboids (speed).")
    ap.add_argument("--det_stride", type=int, default=8, help="Stride sampling for detectors.")
    ap.add_argument("--out_png", default="geometry_3d.png")
    args = ap.parse_args()

    img = read_f32(args.param_image)
    nx, ny, nz = map(int, img[:3])
    vx, vy, vz = img[3:6]
    fov2coll = float(img[11])

    col = read_f32(args.param_collimator)
    num_layers = int(col[0])
    assert num_layers >= 1
    L0 = 10
    num_holes = int(col[L0+0])
    plate_wx = float(col[L0+1])
    plate_th = float(col[L0+2])
    plate_hz = float(col[L0+3])
    plate_y0, plate_y1 = 0.0, plate_th

    det = read_f32(args.param_detector)
    num_det = int(det[0])
    D = det[1:1+12*num_det].reshape(num_det, 12)

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    # Plate as one cuboid (solid)
    plate_cy = (plate_y0 + plate_y1)/2
    plate_faces = cuboid_faces(0.0, plate_cy, 0.0, plate_wx, plate_th, plate_hz)
    ax.add_collection3d(Poly3DCollection(plate_faces, alpha=0.15))

    # Plot detectors as cuboids (sampled)
    faces_all = []
    count = 0
    for i in range(0, num_det, args.det_stride):
        if count >= args.max_det:
            break
        x,y,z, wx,ty,hz = D[i,0], D[i,1], D[i,2], D[i,3], D[i,4], D[i,5]
        faces_all.extend(cuboid_faces(x,y,z, wx,ty,hz))
        count += 1
    ax.add_collection3d(Poly3DCollection(faces_all, alpha=0.05))

    # FOV slab box (based on nx,ny,nz)
    fov_sx, fov_sy, fov_sz = nx*vx, ny*vy, nz*vz
    fov_cx, fov_cy, fov_cz = 0.0, -fov2coll, 0.0
    fov_faces = cuboid_faces(fov_cx, fov_cy, fov_cz, fov_sx, fov_sy, fov_sz)
    ax.add_collection3d(Poly3DCollection(fov_faces, alpha=0.10))

    ax.set_title(f"SCGC Geometry (plate + sampled detectors) | numDet={num_det}, holes={num_holes}\n"
                 f"FOV: ({nx},{ny},{nz}) at y=-{fov2coll}mm")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm) depth")
    ax.set_zlabel("Z (mm)")

    # autoscale
    all_x = D[:,0]; all_y = D[:,1]; all_z = D[:,2]
    ax.set_xlim(float(all_x.min()-20), float(all_x.max()+20))
    ax.set_ylim(-fov2coll-20, float(all_y.max()+20))
    ax.set_zlim(float(all_z.min()-20), float(all_z.max()+20))

    plt.tight_layout()
    plt.savefig(args.out_png, dpi=200)
    print(f"Wrote {args.out_png}")

if __name__ == "__main__":
    main()
