"""
Sound Type Performance Test
Tests DoA algorithm performance across different sound types.
Fixed servo position, varying sound types.
"""
import serial
import serial.tools.list_ports
import time
import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
from my_algos import get_gcc_phat_angle, get_srp_phat_angle, get_basic_cc_angle, get_music_angle

# =============================================================================
# CONFIGURATION
# =============================================================================
COM_PORT = None
BAUD_RATE = 9600
SAMPLE_RATE = 16000
RECORDING_DURATION = 1.0
TEST_ANGLE = 45              # Fixed servo position for all tests
REPETITIONS = 5              # How many times to test each sound type

# Sound types (code, name, description)
SOUND_TYPES = [
    ('0', 'Chirp', 'Broadband sweep 500-3000Hz'),
    ('1', '500Hz', 'Pure tone 500Hz'),
    ('2', '1000Hz', 'Pure tone 1000Hz'),
    ('3', '2000Hz', 'Pure tone 2000Hz'),
    ('4', '4000Hz', 'Pure tone 4000Hz'),
    ('5', 'Noise', 'White noise simulation'),
    ('6', 'Voice', 'Voice-like formants'),
    ('7', 'Click', 'Impulsive click'),
]

# Algorithms
ALGORITHMS = [
    ("GCC-PHAT", get_gcc_phat_angle, "#00d2ff"),
    ("SRP-PHAT", get_srp_phat_angle, "#4ecdc4"),
    ("Basic CC", get_basic_cc_angle, "#ff6b6b"),
    ("MUSIC", get_music_angle, "#ffe66d"),
]

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def send_goto(ser, angle):
    """Move servo to angle."""
    cmd = f"G{angle:03d}"
    ser.write(cmd.encode())
    while True:
        line = ser.readline().decode().strip()
        if line == "READY":
            return

def send_sound(ser, sound_code):
    """Play a specific sound type."""
    cmd = f"S{sound_code}"
    ser.write(cmd.encode())
    # Don't wait - we're recording

# =============================================================================
# MAIN
# =============================================================================

print("=" * 60)
print(" Sound Type Performance Test")
print("=" * 60)

# Connect to Arduino
print("\n[1] Detecting Arduino...")
ports = list(serial.tools.list_ports.comports())
if COM_PORT is None:
    for p in ports:
        if any(x in p.description.lower() for x in ['arduino', 'ch340', 'usb serial']):
            COM_PORT = p.device
            break
    if COM_PORT is None and ports:
        COM_PORT = ports[0].device

if not COM_PORT:
    print("ERROR: No COM ports found!")
    exit(1)

try:
    ard = serial.Serial(COM_PORT, BAUD_RATE, timeout=2)
    time.sleep(2)
    ard.read_all()
    print(f"    Connected to {COM_PORT}")
except Exception as e:
    print(f"ERROR: {e}")
    exit(1)

# Connect to ReSpeaker
print("\n[2] Detecting ReSpeaker...")
respeaker_id = None
for i, dev in enumerate(sd.query_devices()):
    if dev['max_input_channels'] >= 4:
        name = dev['name'].lower()
        if 'respeaker' in name or 'uac1.0' in name or 'seeed' in name:
            respeaker_id = i
            print(f"    Found: {dev['name']}")
            break

if respeaker_id is None:
    print("ERROR: ReSpeaker not found!")
    ard.close()
    exit(1)

# Move to test position
print(f"\n[3] Moving to test angle ({TEST_ANGLE}°)...")
send_goto(ard, TEST_ANGLE)
print("    OK")

# Capture reference angles
print("\n[4] Capturing reference angles...")
ref_recording = sd.rec(
    int(RECORDING_DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=6,
    device=respeaker_id,
    dtype='int16'
)
time.sleep(0.1)
send_sound(ard, '0')  # Use chirp for reference
sd.wait()
ref_audio = ref_recording[:, 1:5].astype(np.float64)

reference_angles = {}
for name, algo_func, _ in ALGORITHMS:
    try:
        reference_angles[name] = algo_func(ref_audio, SAMPLE_RATE)
    except:
        reference_angles[name] = 0
print(f"    References: {', '.join([f'{n[:3]}={v:.0f}°' for n, v in reference_angles.items()])}")

# =============================================================================
# RUN TESTS
# =============================================================================

print("\n" + "=" * 60)
print(" Running Sound Type Tests")
print("=" * 60)
print(f"    Sound types: {len(SOUND_TYPES)}")
print(f"    Repetitions: {REPETITIONS}")
print(f"    Algorithms: {len(ALGORITHMS)}")
print(f"    Total measurements: {len(SOUND_TYPES) * REPETITIONS}")

# Results: {sound_name: {algo_name: [errors]}}
results = {s[1]: {a[0]: [] for a in ALGORITHMS} for s in SOUND_TYPES}

try:
    for sound_code, sound_name, sound_desc in SOUND_TYPES:
        print(f"\n--- Testing: {sound_name} ({sound_desc}) ---")
        
        for rep in range(REPETITIONS):
            print(f"    Rep {rep+1}/{REPETITIONS}: ", end="", flush=True)
            
            # Record
            recording = sd.rec(
                int(RECORDING_DURATION * SAMPLE_RATE),
                samplerate=SAMPLE_RATE,
                channels=6,
                device=respeaker_id,
                dtype='int16'
            )
            time.sleep(0.1)
            send_sound(ard, sound_code)
            sd.wait()
            
            raw_audio = recording[:, 1:5].astype(np.float64)
            
            # Calculate with each algorithm
            for algo_name, algo_func, _ in ALGORITHMS:
                try:
                    raw_est = algo_func(raw_audio, SAMPLE_RATE)
                    ref = reference_angles[algo_name]
                    est = (raw_est - ref + 360) % 360
                    if est > 180:
                        est = est - 360
                    
                    # Error from expected (0° since we're at reference position)
                    err = abs(est)
                    results[sound_name][algo_name].append(err)
                    print(f"{algo_name[:3]}:{est:+5.1f}° ", end="")
                except:
                    results[sound_name][algo_name].append(None)
                    print(f"{algo_name[:3]}:ERR ", end="")
            print()
            
            time.sleep(0.3)  # Brief pause between reps

except KeyboardInterrupt:
    print("\n\nStopped by user.")

finally:
    ard.close()

# =============================================================================
# ANALYSIS & PLOTTING
# =============================================================================

print("\n" + "=" * 60)
print(" Results Summary")
print("=" * 60)

# Calculate mean errors
mean_errors = {s: {} for s in results}
for sound_name in results:
    for algo_name in results[sound_name]:
        valid = [e for e in results[sound_name][algo_name] if e is not None]
        if valid:
            mean_errors[sound_name][algo_name] = np.mean(valid)
        else:
            mean_errors[sound_name][algo_name] = None

# Print table
print("\nMean Error by Sound Type (degrees):")
print("-" * 70)
header = f"{'Sound Type':12s}"
for algo_name, _, _ in ALGORITHMS:
    header += f" | {algo_name:10s}"
print(header)
print("-" * 70)

for sound_code, sound_name, _ in SOUND_TYPES:
    row = f"{sound_name:12s}"
    for algo_name, _, _ in ALGORITHMS:
        val = mean_errors[sound_name].get(algo_name)
        if val is not None:
            row += f" | {val:10.1f}"
        else:
            row += f" | {'N/A':>10s}"
    print(row)
print("-" * 70)

# Overall average per algorithm
print("\nOverall Average Error:")
for algo_name, _, color in ALGORITHMS:
    all_errors = []
    for sound_name in results:
        valid = [e for e in results[sound_name][algo_name] if e is not None]
        all_errors.extend(valid)
    if all_errors:
        print(f"    {algo_name:12s}: {np.mean(all_errors):.1f}°")

# =============================================================================
# PLOTTING
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("DoA Algorithm Performance by Sound Type", fontsize=16, fontweight='bold')

# Plot 1: Heatmap of mean errors
ax1 = axes[0, 0]
sound_names = [s[1] for s in SOUND_TYPES]
algo_names = [a[0] for a in ALGORITHMS]
heatmap_data = np.array([[mean_errors[s].get(a, np.nan) for a in algo_names] for s in sound_names])
im = ax1.imshow(heatmap_data, cmap='RdYlGn_r', aspect='auto')
ax1.set_xticks(range(len(algo_names)))
ax1.set_xticklabels(algo_names, rotation=45, ha='right')
ax1.set_yticks(range(len(sound_names)))
ax1.set_yticklabels(sound_names)
ax1.set_title('Mean Error Heatmap (lower=better)')
for i in range(len(sound_names)):
    for j in range(len(algo_names)):
        val = heatmap_data[i, j]
        if not np.isnan(val):
            ax1.text(j, i, f'{val:.1f}°', ha='center', va='center', fontsize=9)
plt.colorbar(im, ax=ax1, label='Error (°)')

# Plot 2: Grouped bar chart
ax2 = axes[0, 1]
x = np.arange(len(sound_names))
width = 0.2
for i, (algo_name, _, color) in enumerate(ALGORITHMS):
    vals = [mean_errors[s].get(algo_name, 0) for s in sound_names]
    ax2.bar(x + i*width, vals, width, label=algo_name, color=color, alpha=0.8)
ax2.set_xlabel('Sound Type')
ax2.set_ylabel('Mean Error (°)')
ax2.set_title('Error by Sound Type & Algorithm')
ax2.set_xticks(x + width * 1.5)
ax2.set_xticklabels(sound_names, rotation=45, ha='right')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# Plot 3: Algorithm comparison (overall average)
ax3 = axes[1, 0]
overall_means = []
colors = []
for algo_name, _, color in ALGORITHMS:
    all_errors = []
    for sound_name in results:
        valid = [e for e in results[sound_name][algo_name] if e is not None]
        all_errors.extend(valid)
    overall_means.append(np.mean(all_errors) if all_errors else 0)
    colors.append(color)

bars = ax3.bar(algo_names, overall_means, color=colors, alpha=0.8)
for bar, val in zip(bars, overall_means):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
            f'{val:.1f}°', ha='center', fontsize=11, fontweight='bold')
ax3.set_ylabel('Mean Error (°)')
ax3.set_title('Overall Algorithm Performance (All Sound Types)')
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: Box plot by algorithm
ax4 = axes[1, 1]
all_data = []
all_labels = []
all_colors = []
for algo_name, _, color in ALGORITHMS:
    algo_errors = []
    for sound_name in results:
        valid = [e for e in results[sound_name][algo_name] if e is not None]
        algo_errors.extend(valid)
    if algo_errors:
        all_data.append(algo_errors)
        all_labels.append(algo_name)
        all_colors.append(color)

if all_data:
    bp = ax4.boxplot(all_data, labels=all_labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], all_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
ax4.set_ylabel('Error (°)')
ax4.set_title('Error Distribution by Algorithm')
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('sound_type_performance.png', dpi=150)
print(f"\n    Plot saved: sound_type_performance.png")
plt.show()

print("\nDone.")
