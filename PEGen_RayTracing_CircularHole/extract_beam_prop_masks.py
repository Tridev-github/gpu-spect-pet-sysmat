import numpy as np
import h5py
import os

def extract_3d_beams():
    # ============================================================
    # PATH CONFIG (ONLY CHANGE THIS PART)
    # ============================================================
    BASE_DIR = "/tmp/tridevme"

    filename = os.path.join(BASE_DIR, "PE_SysMat_shift_0.000000_0.000000_0.000000.sysmat")
    prop_file = os.path.join(BASE_DIR, "beams_properties_3d.hdf5")
    mask_file = os.path.join(BASE_DIR, "beams_masks_3d.hdf5")

    # ============================================================
    # DIMENSIONS
    # ============================================================
    num_rot = 60
    num_det = 5632
    num_vox_per_det = 4096  # 16x16x16
    total_elements = 1384120320

    # ============================================================
    # OPEN DATA
    # ============================================================
    print("Opening sysmat from:", filename)
    data_1d = np.memmap(filename, dtype='float32', mode='r', shape=(total_elements,))

    # ============================================================
    # OUTPUT FILES
    # ============================================================
    prop_h5 = h5py.File(prop_file, "w")
    mask_h5 = h5py.File(mask_file, "w")

    prop_ds = prop_h5.create_dataset(
        "beam_properties", (0, 11), maxshape=(None, 11), dtype='float32'
    )

    mask_ds = mask_h5.create_dataset(
        "beam_mask",
        (num_rot * num_det, num_vox_per_det),
        dtype='int8'
    )

    print("Extracting 3D beam properties (16x16x16 resolution)...")

    # ============================================================
    # MAIN LOOP
    # ============================================================
    beam_counter = 0

    for r in range(num_rot):
        angle_rad = r * (2 * np.pi / num_rot)

        for d in range(num_det):

            start_idx = (r * num_det * num_vox_per_det) + (d * num_vox_per_det)
            end_idx = start_idx + num_vox_per_det

            ppdf = data_1d[start_idx:end_idx]

            # Threshold
            mx = np.max(ppdf)
            threshold = 0.05 * mx if mx > 0 else 1.0
            mask = (ppdf > threshold).astype('int8')

            # Save mask
            global_idx = r * num_det + d
            mask_ds[global_idx, :] = mask

            # Save properties
            if np.sum(mask) > 0:
                properties = [
                    float(r),
                    float(d),
                    1.0,
                    angle_rad,
                    8.0,
                    float(np.sum(mask)),
                    1.0,
                    float(np.sum(ppdf)),
                    0.0, 0.0, 0.0
                ]

                prop_ds.resize((beam_counter + 1, 11))
                prop_ds[beam_counter] = properties
                beam_counter += 1

        print(f"  - Finished Rotation {r}/60 ({((r+1)/60)*100:.1f}%)")

    prop_h5.close()
    mask_h5.close()

    print("\nExtraction Complete!")
    print("Saved files:")
    print("  ", prop_file)
    print("  ", mask_file)
    print(f"Total active beams: {beam_counter}")


if __name__ == "__main__":
    extract_3d_beams()
