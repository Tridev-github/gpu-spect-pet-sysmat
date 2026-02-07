#!/usr/bin/env python3
import argparse, numpy as np, os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--param_detector", required=True)
    args = ap.parse_args()

    det = np.fromfile(args.param_detector, np.float32)
    n = int(round(det[0]))
    body = det[1:1+n*12].reshape(n,12)

    xyz = body[:,0:3]
    size = body[:,3:6]

    # group by y plane
    y = np.round(xyz[:,1], 3)
    ys = np.unique(y)

    print(f"File={args.param_detector} floats={det.size} bytes={os.path.getsize(args.param_detector)}")
    print(f"numDet={n}")
    print("Y-planes:")
    for yy in sorted(ys):
        idx = np.where(y==yy)[0]
        xs = xyz[idx,0]; zs = xyz[idx,2]
        s = np.unique(np.round(size[idx],3), axis=0)
        print(f"  y={yy:.3f} count={idx.size}  Xspan={xs.max()-xs.min():.3f}  Zspan={zs.max()-zs.min():.3f}  sizes={s.tolist()}")

if __name__ == "__main__":
    main()
