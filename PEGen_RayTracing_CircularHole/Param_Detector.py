import numpy as np

# 1. Create the array of 80,000 zeros (matching the C++ size)
det = np.zeros(80000, dtype=np.float32)

# 2. Total number of detectors (3 layers of 512 + 1 layer of 4096 = 5632)
total_bins = 5632
det[0] = total_bins

count = 0
current_y = 10.0  # Starting 10mm behind the collimator reference

# 3. Loop through the 4 Layers
for layer_id in range(4):
    if layer_id < 3: 
        # Layers 1, 2, and 3 (Mosaic layers)
        nx, nz = 32, 16
        dx, dy, dz = 3.0, 3.0, 6.0  # Width=3, Thickness=3, Height=6
    else:
        # Layer 4 (High Resolution layer)
        nx, nz = 64, 64
        dx, dy, dz = 2.0, 2.0, 6.0  # Width=2, Thickness=2, Height=6
    
    # Calculate the Y-center of the current layer
    center_y = current_y + (dy / 2.0)
    
    for r in range(nz):
        for c in range(nx):
            # Calculate the starting slot for this specific detector
            # Documentation: id_Detector * 12 + 1
            base = count * 12 + 1
            
            # Position centering: Centering the grid around (X=0, Z=0)
            pos_x = (c - (nx - 1) / 2.0) * dx
            pos_z = (r - (nz - 1) / 2.0) * dz
            
            # Mapping values to the slots defined in your documentation
            det[base + 0] = pos_x       # [id*12+1] x center
            det[base + 1] = center_y    # [id*12+2] y center
            det[base + 2] = pos_z       # [id*12+3] z center
            det[base + 3] = dx          # [id*12+4] width
            det[base + 4] = dy          # [id*12+5] thickness
            det[base + 5] = dz          # [id*12+6] height
            
            # Physics: GAGG(Ce) Crystal values for 140keV
            det[base + 6] = 0.35        # [id*12+7] total attenuation
            det[base + 7] = 0.30        # [id*12+8] photo-electric
            det[base + 8] = 0.05        # [id*12+9] compton
            det[base + 9] = 0.10        # [id*12+10] energy resolution (10%)
            det[base + 10] = 0.0        # [id*12+11] rotation angle (y)
            det[base + 11] = 1.0        # [id*12+12] flag (1=active)
            
            count += 1
            
    # Move the starting Y position for the next layer so they stack
    current_y += dy

# 4. Save to the file the C++ code expects
det.tofile("Params_Detector.dat")

print(f"Successfully created Params_Detector.dat")
print(f"Total Detectors: {count}")
print(f"File size: {det.nbytes} bytes")
