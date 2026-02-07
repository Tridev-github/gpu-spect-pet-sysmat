#!/usr/bin/env python3
import argparse, os
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--param_detector", required=True)
    args = ap.parse_args()

    path = args.param_detector
    det = np.fromfile(path, dtype=np.float32)
    nf = det.size
    print(f"File: {path}")
    print(f"Floats: {nf}  Bytes: {os.path.getsize(path)}")

    if nf < 1:
        raise RuntimeError("Empty detector file.")

    num_det = int(round(det[0]))
    print(f"Header numDet = {num_det}")

    expected = 1 + 12*num_det
    print(f"Expected floats = {expected}")

    if nf == expected:
        print("OK: file length matches header.")
        D = det[1:].reshape(num_det, 12)
    elif nf > expected:
        print(f"WARNING: file has EXTRA floats: {nf-expected}.")
        print("I will IGNORE the tail and parse the first expected floats.")
        D = det[1:expected].reshape(num_det, 12)
        tail = det[expected:expected+8]
        print(f"Tail sample (first 8 extra floats) = {tail.tolist()}")
    else:
        print(f"ERROR: file is TOO SHORT by {expected-nf} floats.")
        print("This file cannot be parsed correctly. Regenerate it.")
        return

    xyz = D[:, 0:3]
    size = D[:, 3:6]

    mn = xyz.min(axis=0); mx = xyz.max(axis=0)
    print(f"XYZ bbox min = {mn.tolist()}")
    print(f"XYZ bbox max = {mx.tolist()}")
    print(f"Unique sizes (rounded 1e-3): {np.unique(np.round(size,3), axis=0).tolist()}")

    y = np.round(xyz[:,1], 3)
    uy, cnt = np.unique(y, return_counts=True)
    order = np.argsort(cnt)[::-1][:10]
    print("Top Y planes (rounded 0.001):")
    for i in order:
        print(f"  y={uy[i]:.3f} count={cnt[i]}")

if __name__ == "__main__":
    main()
