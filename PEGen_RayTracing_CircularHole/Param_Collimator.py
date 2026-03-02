import numpy as np
import random

# 1. Create an array of 80,000 zeros
col = np.zeros(80000, dtype=np.float32)

# 2. Set numCollimatorLayers
col[0] = 1 

# 3. Set Metadata for Layer 0 (Indices: (0+1)*10 + offset)
id_layer = 0
base_layer = (id_layer + 1) * 10

num_holes = 1865
width_x = 200.0
thickness_y = 5.0
height_z = 150.0

col[base_layer + 0] = num_holes    # Number of holes
col[base_layer + 1] = width_x      # Width (X)
col[base_layer + 2] = thickness_y  # Thickness (Y)
col[base_layer + 3] = height_z     # Height (Z)
col[base_layer + 4] = 0.0          # Distance between 1st layer and this layer
col[base_layer + 5] = 4.0          # Total Attenuation (Tungsten)
col[base_layer + 6] = 3.8          # PE Attenuation
col[base_layer + 7] = 0.2          # Compton Attenuation

# 4. Generate Random Holes
# Radius is half of the diameter (1.6mm / 2 = 0.8mm)
radius = 0.8

for i in range(num_holes):
    # Formula: id_Hole * 9 + 100
    base_hole = i * 9 + 100
    
    # Randomly place holes within the plate dimensions
    # We subtract the radius to ensure holes don't go over the edge
    pos_x = random.uniform(-(width_x/2) + radius, (width_x/2) - radius)
    pos_z = random.uniform(-(height_z/2) + radius, (height_z/2) - radius)
    
    col[base_hole + 0] = pos_x       # x center
    col[base_hole + 1] = 0.0         # y1 center (start of hole)
    col[base_hole + 2] = thickness_y # y2 center (end of hole)
    col[base_hole + 3] = pos_z       # z center
    col[base_hole + 4] = radius      # R (Radius)
    
    # Hole interior is air, so attenuation is 0
    col[base_hole + 5] = 0.0         # Total Attenuation
    col[base_hole + 6] = 0.0         # PE Attenuation
    col[base_hole + 7] = 0.0         # Compton Attenuation
    
    col[base_hole + 8] = 1.0         # Flag (1 = active)

# 5. Save with the name the C++ code expects
col.tofile("Params_Collimator.dat")

print(f"Successfully created Params_Collimator.dat with {num_holes} holes.")
