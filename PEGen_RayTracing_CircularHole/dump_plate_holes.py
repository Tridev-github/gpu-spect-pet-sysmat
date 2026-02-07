#!/usr/bin/env python3
import argparse, numpy as np, os, math

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--param_collimator", required=True)
    args = ap.parse_args()

    col = np.fromfile(args.param_collimator, np.float32)
    nl = int(round(col[0]))
    print(f"File={args.param_collimator} floats={col.size} bytes={os.path.getsize(args.param_collimator)}")
    print(f"numLayers={nl}")
    for lid in range(nl):
        base = (lid+1)*10
        nh = int(round(col[base+0]))
        wX = float(col[base+1])
        tY = float(col[base+2])
        hZ = float(col[base+3])
        dist = float(col[base+4])
        print(f"Layer{lid}: holes={nh} widthX={wX} thickY={tY} heightZ={hZ} dist={dist}")
        holes = []
        for hid in range(nh):
            off=100+hid*9
            x,y1,y2,z,r = map(float, col[off:off+5])
            holes.append((x,y1,y2,z,r))
        holes=np.array(holes, np.float32)
        r = holes[:,4]
        # open fraction estimate
        plate_area = wX*hZ
        open_area = nh*math.pi*(r.mean()**2)
        of = open_area/plate_area
        print(f"  hole radius mean={r.mean():.4f} (diam={2*r.mean():.4f})")
        print(f"  y1 range=({holes[:,1].min():.3f},{holes[:,1].max():.3f})  y2 range=({holes[:,2].min():.3f},{holes[:,2].max():.3f})")
        print(f"  X range=({holes[:,0].min():.3f},{holes[:,0].max():.3f})  Z range=({holes[:,3].min():.3f},{holes[:,3].max():.3f})")
        print(f"  approx open fraction={of*100:.2f}% (paper=12.5%)")
        print("  first 5 holes:")
        for i in range(min(5, nh)):
            x,y1,y2,z,r = holes[i]
            print(f"    {i}: x={x:.3f} z={z:.3f} y1={y1:.3f} y2={y2:.3f} r={r:.3f}")

if __name__ == "__main__":
    main()

