"""
ReSpeaker Hardware DoA Visualization
Reads Direction of Arrival from the device firmware and displays it in real-time
"""
from tuning import Tuning
import usb.core
import usb.util
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Wedge, Circle
import numpy as np

# Find device
dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)

if not dev:
    print("ERROR: ReSpeaker device not found!")
    exit(1)

# Initialize tuning interface
Mic_tuning = Tuning(dev)

# Test initial read
print(f"Initial DoA: {Mic_tuning.direction}°")
print("Starting visualization...")

# Visualization setup
plt.style.use('dark_background')
fig = plt.figure(figsize=(10, 10))
fig.patch.set_facecolor('#0a0a0a')

ax = fig.add_subplot(111, projection='polar')
ax.set_facecolor('#0a0a0a')
ax.set_title("ReSpeaker Hardware DoA", color='white', fontsize=18, 
             fontweight='bold', pad=20)

# Configure polar plot
ax.set_theta_zero_location('N')  # 0° at top
ax.set_theta_direction(-1)  # Clockwise
ax.set_ylim(0, 1.5)
ax.set_yticks([])
ax.grid(True, color='#333333', linestyle=':', linewidth=1, alpha=0.5)
ax.set_xticks(np.radians([0, 45, 90, 135, 180, 225, 270, 315]))
ax.set_xticklabels(['FRONT\n0°', 'FR\n45°', 'RIGHT\n90°', 'BR\n135°', 
                    'BACK\n180°', 'BL\n225°', 'LEFT\n270°', 'FL\n315°'], 
                   color='#888888', fontsize=11)

# Draw microphone positions (ReSpeaker v2.0 layout)
mic_angles = [45, 315, 225, 135]  # M0, M1, M2, M3
mic_labels = ['M0', 'M1', 'M2', 'M3']
for i, (angle, label) in enumerate(zip(mic_angles, mic_labels)):
    ax.plot([np.radians(angle)], [0.35], 'o', color='#00ff88', 
           markersize=12, markeredgecolor='white', markeredgewidth=2, zorder=10)
    ax.text(np.radians(angle), 0.5, label, ha='center', va='center',
           color='#00ff88', fontsize=10, fontweight='bold')

# Center circle (array outline)
center_circle = Circle((0, 0), 0.4, transform=ax.transData._b,
                      fill=False, edgecolor='#00ff88', linewidth=2, alpha=0.3)
ax.add_patch(center_circle)

# Direction indicator (wedge)
wedge = Wedge((0, 0), 1.3, 0, 30, width=0.3,
             transform=ax.transData._b,
             facecolor='#00d2ff', edgecolor='#ffffff',
             linewidth=2, alpha=0.7, zorder=5)
ax.add_patch(wedge)

# Direction line
direction_line, = ax.plot([], [], color='#00d2ff', linewidth=4, zorder=8)

# Angle text
angle_text = ax.text(0.5, 0.5, '0°', transform=ax.transAxes,
                    ha='center', va='center', color='#00d2ff',
                    fontsize=32, fontweight='bold')

# History for smoothing
angle_history = []
history_size = 5

def smooth_angle(new_angle):
    """Circular averaging for smooth display."""
    global angle_history
    
    angle_history.append(new_angle)
    if len(angle_history) > history_size:
        angle_history.pop(0)
    
    # Circular mean
    angles_rad = np.radians(angle_history)
    mean_sin = np.mean(np.sin(angles_rad))
    mean_cos = np.mean(np.cos(angles_rad))
    smooth = np.degrees(np.arctan2(mean_sin, mean_cos))
    
    return (smooth + 360) % 360

def update(frame):
    """Update visualization with new DoA reading."""
    try:
        # Read DoA from device
        angle = Mic_tuning.direction
        
        if angle is None:
            return wedge, direction_line, angle_text
        
        # Smooth the angle
        smooth_angle_val = smooth_angle(angle)
        
        # Update wedge (beam)
        beam_width = 30
        wedge.set_theta1(smooth_angle_val - beam_width/2)
        wedge.set_theta2(smooth_angle_val + beam_width/2)
        
        # Update direction line
        angle_rad = np.radians(smooth_angle_val)
        direction_line.set_data([angle_rad, angle_rad], [0, 1.3])
        
        # Update text
        angle_text.set_text(f"{smooth_angle_val:.0f}°")
        
        return wedge, direction_line, angle_text
        
    except Exception as e:
        print(f"Error reading DoA: {e}")
        return wedge, direction_line, angle_text

# Create animation
ani = animation.FuncAnimation(fig, update, interval=100, 
                             blit=False, cache_frame_data=False)

print("Visualization running. Close window to exit.")

try:
    plt.show()
except KeyboardInterrupt:
    pass

print("\nStopped.")
Mic_tuning.close()