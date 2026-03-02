import numpy as np
import matplotlib.pyplot as plt

def analyze_3d_final():
    # 1. PARAMETERS (16x16x16 3D Grid)
    nx, ny, nz = 16, 16, 16
    vx_size = 8.0 # mm (128mm / 16 voxels)
    num_rot, num_det = 60, 5632
    filename = "PE_SysMat_shift_0.000000_0.000000_0.000000.sysmat"

    # 2. OPEN DATA (Memory Mapping for Efficiency)
    data = np.memmap(filename, dtype='float32', mode='r', 
                     shape=(num_rot, num_det, nx*ny*nz))

    # 3. CREATE FIGURE WITH CONSTRAINED LAYOUT (No Cutoff)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), constrained_layout=True)

    # ---------------------------------------------------------
    # PANEL 1: PPDF (View from One Crystal)
    # ---------------------------------------------------------
    target_det = 2800
    ppdf_3d = data[0, target_det, :].reshape(nx, ny, nz)
    ppdf_mip = np.max(ppdf_3d, axis=2) # 3D to 2D projection
    
    # extent sets the axis to -64 to 64 mm
    extent = [-64, 64, -64, 64]
    
    # Using 'viridis' colormap to match the Sensitivity Map
    im1 = ax1.imshow(ppdf_mip, cmap='viridis', origin='lower', extent=extent)
    ax1.set_title(f"Detector {target_det}: PPDF\n(Individual Detection Probability)", fontsize=14, fontweight='bold')
    ax1.set_xlabel("X Position (mm)")
    ax1.set_ylabel("Z Position (mm)")
    fig.colorbar(im1, ax=ax1, label="Probability") # UPDATED LABEL

    # ---------------------------------------------------------
    # PANEL 2: SENSITIVITY MAP (Total System View)
    # ---------------------------------------------------------
    print("Calculating Sensitivity Map from all detectors and rotations...")
    sensitivity_3d = np.sum(data, axis=(0, 1)).reshape(nx, ny, nz)
    sens_slice = sensitivity_3d[:, :, 8] # View the middle slice
    
    im2 = ax2.imshow(sens_slice, cmap='viridis', origin='lower', extent=extent)
    ax2.set_title("Total System Sensitivity)", fontsize=14, fontweight='bold')
    ax2.set_xlabel("X Position (mm)")
    ax2.set_ylabel("Z Position (mm)")
    fig.colorbar(im2, ax=ax2, label="Sensitivity") # UPDATED LABEL

    # ---------------------------------------------------------
    # OVERALL TITLE & FILENAME
    # ---------------------------------------------------------
    #plt.suptitle(f"3D SC-SPECT System Matrix Analysis: 16x16x16 Grid\n(Voxel Size: 8mm x 8mm)", 
                 #fontsize=18, y=0.98)

    output_filename = "verified_system_analysis_16x16x16.png"
    plt.savefig(output_filename, dpi=300)
    print(f"Success! Image saved as {output_filename}")
    plt.show()

if __name__ == "__main__":
    analyze_3d_final()
