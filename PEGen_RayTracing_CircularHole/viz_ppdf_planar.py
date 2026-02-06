#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import argparse

def read_f32(path):
    return np.fromfile(path, dtype=np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sysmat", required=True)
    ap.add_argument("--param_detector", default="Param_Detector.dat")
    ap.add_argument("--param_image", default="Param_Image.dat")
    ap.add_argument("--det_id", type=int, default=0)
    ap.add_argument("--out", default="ppdf.png")
    args = ap.parse_args()

    # image info
    img = read_f32(args.param_image)
    nx, ny, nz = int(img[0]), int(img[1]), int(img[2])

    # detector info
    det = read_f32(args.param_detector)
    num_det = int(det[0])

    # sysmat
    sm = read_f32(args.sysmat)

    plane_vox = nx * ny
    expected = num_det * plane_vox

    print("detectors:", num_det)
    print("plane voxels:", plane_vox)
    print("file floats:", sm.size)

    if sm.size != expected:
        print("WARNING: file size does not match expected planar SM")

    sm = sm.reshape((num_det, plane_vox))

    ppdf = sm[args.det_id].reshape((ny, nx))

    print("PPDF stats:")
    print("min:", np.min(ppdf))
    print("max:", np.max(ppdf))
    print("nonzero:", np.count_nonzero(ppdf))

    plt.figure(figsize=(6,6))
    plt.imshow(ppdf, origin="lower")
    plt.colorbar()
    plt.title(f"Detector {args.det_id} planar PPDF")
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print("Saved:", args.out)

if __name__ == "__main__":
    main()
