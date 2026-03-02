import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def draw_block(ax, center, size, color, alpha=0.8, edge=True):
    """Draws a 3D block (crystal/FOV/plate)"""
    x, y, z = center
    dx, dy, dz = size
    xx = [x-dx/2, x+dx/2]; yy = [y-dy/2, y+dy/2]; zz = [z-dz/2, z+dz/2]
    faces = [
        [[xx[0],yy[0],zz[0]], [xx[1],yy[0],zz[0]], [xx[1],yy[1],zz[0]], [xx[0],yy[1],zz[0]]],
        [[xx[0],yy[0],zz[1]], [xx[1],yy[0],zz[1]], [xx[1],yy[1],zz[1]], [xx[0],yy[1],zz[1]]],
        [[xx[0],yy[0],zz[0]], [xx[1],yy[0],zz[0]], [xx[1],yy[0],zz[1]], [xx[0],yy[0],zz[1]]],
        [[xx[0],yy[1],zz[0]], [xx[1],yy[1],zz[0]], [xx[1],yy[1],zz[1]], [xx[0],yy[1],zz[1]]],
        [[xx[0],yy[0],zz[0]], [xx[0],yy[1],zz[0]], [xx[0],yy[1],zz[1]], [xx[0],yy[0],zz[1]]],
        [[xx[1],yy[0],zz[0]], [xx[1],yy[1],zz[0]], [xx[1],yy[1],zz[1]], [xx[1],yy[0],zz[1]]]
    ]
    ec = 'black' if edge else 'none'
    ax.add_collection3d(Poly3DCollection(faces, facecolors=color, edgecolors=ec, alpha=alpha, linewidths=0.1))

def generate_final_report():
    # 1. LOAD ALL DATA
    det_raw = np.fromfile("Params_Detector.dat", dtype=np.float32)
    col_raw = np.fromfile("Params_Collimator.dat", dtype=np.float32)
    img_raw = np.fromfile("Params_Image.dat", dtype=np.float32)
    
    det_data = det_raw[1:1 + 5632*12].reshape(-1, 12)
    hole_data = col_raw[100:100 + 1865*9].reshape(-1, 9)
    fov_dist = img_raw[11]
    fov_w = img_raw[0] * img_raw[3]

    fig = plt.figure(figsize=(26, 12))
    layer_colors = ['#aec7e8', '#7fb3d5', '#1f77b4', '#08306b']
    indices = [0, 512, 1024, 1536, 5632]

    # ---------------------------------------------------------
    # PANEL 1: SIDE VIEW (Detailed Shapes)
    # ---------------------------------------------------------
    ax1 = fig.add_subplot(131)
    ax1.set_title("1. SIDE VIEW (Hardware Shapes)", fontsize=14, fontweight='bold')
    
    # Draw FOV as a thin vertical rectangle (2mm thick)
    ax1.add_patch(Rectangle((-fov_dist-1, -fov_w/2), 2, fov_w, color='green', alpha=0.3, label='FOV'))
    
    # Draw Collimator as a solid grey rectangle (5mm thick)
    ax1.add_patch(Rectangle((0, -100), 5, 200, color='grey', alpha=0.5, label='Tungsten Plate'))
    # Show holes as gaps (white dots) in the grey plate
    ax1.scatter(np.random.uniform(0.5, 4.5, 200), np.random.uniform(-100, 100, 200), s=1, color='white')

    # Draw Detector Crystals as Squares
    for i in range(4):
        layer = det_data[indices[i]:indices[i+1]]
        # We plot a sample of 200 crystals per layer for clarity
        sample = layer[::max(1, len(layer)//200)]
        ax1.scatter(sample[:, 1], sample[:, 0], s=8, marker='s', color=layer_colors[i], edgecolors='black', linewidths=0.2, label=f'Layer {i+1}')

    ax1.set_xlabel("Y - Depth/Distance (mm)"); ax1.set_ylabel("X - Width (mm)")
    ax1.set_xlim(-120, 40); ax1.set_ylim(-110, 110)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper left', fontsize=8)


    # ---------------------------------------------------------
    # PANEL 2: LITERAL 3D ASSEMBLY (Physical Reality)
    # ---------------------------------------------------------
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.set_title("2. LITERAL 3D ASSEMBLY", fontsize=14, fontweight='bold')
    draw_block(ax2, [0, -fov_dist, 0], [fov_w, 2, fov_w], 'green', alpha=0.1)
    draw_block(ax2, [0, 2.5, 0], [200, 5, 150], 'grey', alpha=0.2)
    for i in range(4):
        layer = det_data[indices[i]:indices[i+1]]
        step = 15 if i < 3 else 50
        for d in layer[::step]:
            draw_block(ax2, [d[0], d[1], d[2]], [d[3], d[4], d[5]], layer_colors[i], alpha=0.6)
    ax2.set_xlabel('X'); ax2.set_ylabel('Y (Depth)'); ax2.set_zlabel('Z')
    ax2.set_xlim(-110, 110); ax2.set_ylim(-110, 30); ax2.set_zlim(-80, 80)
    ax2.view_init(elev=20, azim=-45)

    # ---------------------------------------------------------
    # PANEL 3: EXPLODED VIEW (The Design Anatomy)
    # ---------------------------------------------------------
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.set_title("3. EXPLODED VIEW", fontsize=14, fontweight='bold')
    
    # Define vertical heights for exploded look
    z_exp = [150, 100, 50, 0] 
    label_x = -240 # Shift labels to the far LEFT side

    # A. Draw FOV (Patient) at the very top
    draw_block(ax3, [0, 0, 250], [fov_w, fov_w, 5], 'green', alpha=0.15)
    ax3.text(label_x, 0, 250, "FOV ", color='green', fontweight='bold', ha='right')

    # B. Draw Collimator Plate
    draw_block(ax3, [0, 0, 200], [200, 150, 5], 'grey', alpha=0.25)
    ax3.scatter(hole_data[::2, 0], hole_data[::2, 3], 200, s=0.5, color='black', alpha=0.4)
    ax3.text(label_x, 0, 200, "COLLIMATOR", color='black', fontweight='bold', ha='right')

    # C. Draw Detector Layers
    for i in range(4):
        layer = det_data[indices[i]:indices[i+1]]
        step = 10 if i < 3 else 35
        for d in layer[::step]:
            draw_block(ax3, [d[0], d[2], z_exp[i]], [d[3], d[5], d[4]], layer_colors[i])
        ax3.text(label_x, 0, z_exp[i], f"Layer {i+1}", color=layer_colors[i], fontweight='bold', ha='right')

    ax3.set_axis_off(); ax3.set_box_aspect((1,1,1.8))
    ax3.view_init(elev=15, azim=30)

    plt.suptitle(f"SC-SPECT System Hardware Verification Report", fontsize=20, y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    plt.savefig("final_scientific_report.png", dpi=300)
    print("Success! 'final_scientific_report.png' created.")
    plt.show()

generate_final_report()
