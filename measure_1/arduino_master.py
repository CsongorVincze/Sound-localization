"""
Arduino Calibration Master - DoA Measurement System
Controls a stepper motor and sound source to measure DoA algorithm accuracy.
Compares multiple algorithms: GCC-PHAT, SRP-PHAT, Basic CC, MUSIC, CNN
"""
import serial
import serial.tools.list_ports
import time
import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
from my_algos import get_gcc_phat_angle, get_srp_phat_angle, get_basic_cc_angle, get_music_angle
from pathlib import Path

# Try to import PyTorch for CNN
CNN_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    CNN_AVAILABLE = True
except ImportError:
    print("WARNING: PyTorch not available. CNN algorithm will be skipped.")

# =============================================================================
# CONFIGURATION
# =============================================================================
COM_PORT = None        # Set to 'COM3' etc. or None for auto-detect
BAUD_RATE = 9600
SAMPLE_RATE = 16000
RECORDING_DURATION = 1.0
STEP_INCREMENT = 5     # degrees
TOTAL_STEPS = 36       # 180 degrees total
DEFAULT_START_ANGLE = 45  # Default servo home position

# CNN Configuration
MIC_SPACING = 0.0465
SPEED_OF_SOUND = 343.0
MAX_TAU = 2 * MIC_SPACING / SPEED_OF_SOUND
CC_LENGTH = 64

# =============================================================================
# CNN MODEL & FEATURE EXTRACTION
# =============================================================================

if CNN_AVAILABLE:
    class DoACNN(nn.Module):
        """CNN for DoA estimation from GCC-PHAT features."""
        def __init__(self):
            super(DoACNN, self).__init__()
            self.conv1 = nn.Conv1d(6, 32, kernel_size=5, padding=2)
            self.bn1 = nn.BatchNorm1d(32)
            self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
            self.bn2 = nn.BatchNorm1d(64)
            self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
            self.bn3 = nn.BatchNorm1d(128)
            self.pool = nn.MaxPool1d(2)
            self.dropout = nn.Dropout(0.3)
            self.fc1 = nn.Linear(128 * 8, 256)
            self.fc2 = nn.Linear(256, 64)
            self.fc3 = nn.Linear(64, 2)
            self.relu = nn.ReLU()
            
        def forward(self, x):
            x = self.pool(self.relu(self.bn1(self.conv1(x))))
            x = self.pool(self.relu(self.bn2(self.conv2(x))))
            x = self.pool(self.relu(self.bn3(self.conv3(x))))
            x = x.view(x.size(0), -1)
            x = self.dropout(self.relu(self.fc1(x)))
            x = self.dropout(self.relu(self.fc2(x)))
            x = self.fc3(x)
            x = x / (torch.norm(x, dim=1, keepdim=True) + 1e-8)
            return x

    def compute_gcc_phat(sig1, sig2, fs, n_output=64):
        """Compute GCC-PHAT correlation vector."""
        n = len(sig1) + len(sig2)
        SIG1 = np.fft.rfft(sig1, n=n)
        SIG2 = np.fft.rfft(sig2, n=n)
        R = SIG1 * np.conj(SIG2)
        mag = np.abs(R)
        R = R / (mag + 1e-10)
        cc = np.fft.irfft(R, n=n)
        max_shift = int(MAX_TAU * fs) + n_output // 2
        cc = np.concatenate([cc[-max_shift:], cc[:max_shift+1]])
        center = len(cc) // 2
        start = center - n_output // 2
        end = start + n_output
        return cc[start:end].astype(np.float32)

    def extract_gcc_features(audio, fs=16000):
        """Extract GCC-PHAT features for all mic pairs."""
        pairs = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
        features = []
        for i, j in pairs:
            cc = compute_gcc_phat(audio[:, i], audio[:, j], fs, CC_LENGTH)
            features.append(cc)
        return np.array(features, dtype=np.float32)

    # Load CNN model
    CNN_MODEL = None
    CNN_DEVICE = torch.device('cpu')
    model_path = Path(__file__).parent / 'doa_cnn_model.pth'
    
    if model_path.exists():
        try:
            CNN_MODEL = DoACNN()
            CNN_MODEL.load_state_dict(torch.load(model_path, map_location=CNN_DEVICE))
            CNN_MODEL.eval()
            print(f"    CNN model loaded from {model_path}")
        except Exception as e:
            print(f"    WARNING: Could not load CNN model: {e}")
            CNN_MODEL = None
    else:
        print(f"    WARNING: CNN model not found at {model_path}")

    def get_cnn_angle(mics, fs=16000):
        """CNN-based DoA estimation."""
        if CNN_MODEL is None:
            return 0.0
        
        features = extract_gcc_features(mics, fs)
        features_tensor = torch.from_numpy(features).unsqueeze(0).float()
        
        with torch.no_grad():
            output = CNN_MODEL(features_tensor)
        
        sin_val = output[0, 0].item()
        cos_val = output[0, 1].item()
        angle = np.degrees(np.arctan2(sin_val, cos_val))
        return (angle + 360) % 360

# Build algorithm list
ALGORITHMS = [
    ("GCC-PHAT", get_gcc_phat_angle, "#00d2ff"),
    ("SRP-PHAT", get_srp_phat_angle, "#4ecdc4"),
    ("Basic CC", get_basic_cc_angle, "#ff6b6b"),
    ("MUSIC", get_music_angle, "#ffe66d"),
]

if CNN_AVAILABLE and 'CNN_MODEL' in dir() and CNN_MODEL is not None:
    ALGORITHMS.append(("CNN", get_cnn_angle, "#9b59b6"))

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def send_goto(ser, angle):
    """Send GOTO command to move servo smoothly to target angle."""
    cmd = f"G{angle:03d}"
    ser.write(cmd.encode())
    # Wait for READY
    while True:
        line = ser.readline().decode().strip()
        if line == "READY":
            return True
        if line == "ERROR":
            return False

def send_play(ser):
    """Send PLAY command to trigger sound."""
    ser.write(b'P')
    # Don't wait - we're recording

def send_reset(ser):
    """Send RESET command to return to 0°."""
    ser.write(b'R')
    while True:
        line = ser.readline().decode().strip()
        if line == "RESET_DONE":
            return

# =============================================================================
# MAIN
# =============================================================================

print("=" * 60)
print(" DoA Algorithm Calibration System")
print("=" * 60)

# 1. Connect to Arduino
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
    ard.read_all()  # Clear buffer
    print(f"    Connected to Arduino on {COM_PORT}")
except Exception as e:
    print(f"ERROR: Could not open {COM_PORT}: {e}")
    exit(1)

# 2. Connect to ReSpeaker
print("\n[2] Detecting ReSpeaker...")
respeaker_id = None
devices = sd.query_devices()

for i, dev in enumerate(devices):
    if dev['max_input_channels'] >= 4:
        name = dev['name'].lower()
        if 'respeaker' in name or 'uac1.0' in name or 'seeed' in name:
            respeaker_id = i
            print(f"    Found ReSpeaker: {dev['name']} (ID: {i})")
            break

if respeaker_id is None:
    print("ERROR: ReSpeaker not found!")
    ard.close()
    exit(1)

# =============================================================================
# ALIGNMENT MODE (Optional)
# =============================================================================

print("\n" + "=" * 60)
print(" Alignment Mode")
print("=" * 60)
print(f"    Default start angle: {DEFAULT_START_ANGLE}°")
print("    Enter an angle (0-180) to test, or:")
print("    - 'skip' to start measurement from default")
print("    - 'done' to start measurement from current position")

current_servo_angle = DEFAULT_START_ANGLE

# Move to default position
print(f"\n    Moving to default position ({DEFAULT_START_ANGLE}°)...", end=" ", flush=True)
send_goto(ard, DEFAULT_START_ANGLE)
print("OK")

try:
    while True:
        user_input = input(f"\n    Servo [{current_servo_angle}°] > ").strip().lower()
        
        if user_input in ['skip', 's']:
            current_servo_angle = DEFAULT_START_ANGLE
            print(f"    Using default: {DEFAULT_START_ANGLE}°")
            break
            
        if user_input in ['done', 'd', 'q', '']:
            print(f"    Starting from {current_servo_angle}°")
            break
            
        try:
            target_angle = int(user_input)
            if target_angle < 0 or target_angle > 180:
                print("    Error: Angle must be 0-180")
                continue
            
            print(f"    Moving to {target_angle}°...", end=" ", flush=True)
            send_goto(ard, target_angle)
            current_servo_angle = target_angle
            print("OK")
            
            # Ping and measure
            print("    Measuring...", end=" ", flush=True)
            recording = sd.rec(
                int(1.0 * SAMPLE_RATE),
                samplerate=SAMPLE_RATE,
                channels=6,
                device=respeaker_id,
                dtype='int16'
            )
            time.sleep(0.1)
            send_play(ard)
            sd.wait()
            
            raw_audio = recording[:, 1:5].astype(np.float64)
            est_angle = get_gcc_phat_angle(raw_audio, SAMPLE_RATE)
            err = est_angle if est_angle < 180 else est_angle - 360
            print(f"DoA: {est_angle:5.1f}° (offset: {err:+5.1f}°)")
            
        except ValueError:
            print("    Invalid input. Enter a number or 'done'/'skip'.")

except KeyboardInterrupt:
    print("\n    Cancelled.")
    ard.close()
    exit(0)

# =============================================================================
# MEASUREMENT LOOP
# =============================================================================

alignment_offset = current_servo_angle

print("\n" + "=" * 60)
print(" Starting Measurement")
print("=" * 60)
print(f"    Start Position: {alignment_offset}°")
print(f"    Steps: {TOTAL_STEPS} x {STEP_INCREMENT}°")
print(f"    Algorithms: {', '.join([a[0] for a in ALGORITHMS])}")

remaining_space = 180 - alignment_offset
max_steps = int(remaining_space / STEP_INCREMENT)
if max_steps < TOTAL_STEPS:
    print(f"    WARNING: Limited to {max_steps} steps (servo limit)")
    TOTAL_STEPS = max_steps

# Results storage: {algorithm_name: [(true, est, err), ...]}
results = {name: [] for name, _, _ in ALGORITHMS}

# First, capture reference angles for absolute algorithms (SRP-PHAT, MUSIC)
# These algorithms report absolute direction, so we need to know what angle
# corresponds to "0° true angle" in their coordinate system
print("\n    Capturing reference angles...", end=" ", flush=True)
ref_recording = sd.rec(
    int(RECORDING_DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=6,
    device=respeaker_id,
    dtype='int16'
)
time.sleep(0.1)
send_play(ard)
sd.wait()
ref_audio = ref_recording[:, 1:5].astype(np.float64)

# Get reference angles for each algorithm
reference_angles = {}
for name, algo_func, _ in ALGORITHMS:
    try:
        reference_angles[name] = algo_func(ref_audio, SAMPLE_RATE)
    except:
        reference_angles[name] = 0
print("OK")
print(f"    Reference angles: " + ", ".join([f"{n[:3]}={v:.0f}°" for n, v in reference_angles.items()]))

try:
    for step in range(TOTAL_STEPS):
        true_angle = step * STEP_INCREMENT
        servo_pos = alignment_offset + true_angle
        
        print(f"\n--- Step {step + 1}/{TOTAL_STEPS} | True: {true_angle}° | Servo: {servo_pos}° ---")
        
        # Move
        print("    Moving...", end=" ", flush=True)
        send_goto(ard, servo_pos)
        current_servo_angle = servo_pos
        print("OK")
        
        # Record
        print("    Recording...", end=" ", flush=True)
        recording = sd.rec(
            int(RECORDING_DURATION * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=6,
            device=respeaker_id,
            dtype='int16'
        )
        time.sleep(0.1)
        send_play(ard)
        sd.wait()
        print("OK")
        
        # Extract mic channels
        raw_audio = recording[:, 1:5].astype(np.float64)
        
        # Calculate with each algorithm
        print("    Calculating: ", end="", flush=True)
        for name, algo_func, _ in ALGORITHMS:
            try:
                raw_est = algo_func(raw_audio, SAMPLE_RATE)
                
                # For absolute algorithms (SRP-PHAT, MUSIC), convert to relative
                # by subtracting the reference angle
                ref = reference_angles[name]
                est = (raw_est - ref + 360) % 360
                
                # Handle wraparound for display (keep in 0-180 range for comparison)
                if est > 180:
                    est = est - 360  # Convert to negative for display
                
                err = abs(est - true_angle)
                if err > 180:
                    err = 360 - err
                    
                results[name].append((true_angle, est, err))
                print(f"{name[:3]}:{est:5.1f}° ", end="")
            except Exception as e:
                results[name].append((true_angle, None, None))
                print(f"{name[:3]}:ERR ", end="")
        print()

except KeyboardInterrupt:
    print("\n\nStopped by user.")

finally:
    ard.close()

# =============================================================================
# RESULTS & PLOTTING
# =============================================================================

print("\n" + "=" * 60)
print(" Results Summary")
print("=" * 60)

# Calculate stats for each algorithm
for name, _, _ in ALGORITHMS:
    valid = [(t, e, err) for t, e, err in results[name] if e is not None]
    if valid:
        errors = [err for _, _, err in valid]
        print(f"    {name:12s}: Mean={np.mean(errors):5.1f}°  Max={np.max(errors):5.1f}°  Std={np.std(errors):4.1f}°")
    else:
        print(f"    {name:12s}: No valid data")

# Create comparison plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("DoA Algorithm Comparison", fontsize=16, fontweight='bold')

# Plot 1: Tracking accuracy for all algorithms
ax1 = axes[0, 0]
for name, _, color in ALGORITHMS:
    valid = [(t, e) for t, e, _ in results[name] if e is not None]
    if valid:
        true_vals = [t for t, _ in valid]
        est_vals = [e for _, e in valid]
        ax1.plot(true_vals, est_vals, 'o-', color=color, label=name, markersize=4)
ax1.plot([0, 180], [0, 180], 'k--', linewidth=2, label='Perfect', alpha=0.5)
ax1.set_xlabel('True Angle (°)')
ax1.set_ylabel('Estimated Angle (°)')
ax1.set_title('Tracking Accuracy')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

# Plot 2: Error comparison
ax2 = axes[0, 1]
bar_width = 0.8 / len(ALGORITHMS)
for i, (name, _, color) in enumerate(ALGORITHMS):
    valid = [(t, err) for t, _, err in results[name] if err is not None]
    if valid:
        x = np.array([t for t, _ in valid]) + i * bar_width
        y = [err for _, err in valid]
        ax2.bar(x, y, width=bar_width, color=color, label=name, alpha=0.7)
ax2.set_xlabel('True Angle (°)')
ax2.set_ylabel('Absolute Error (°)')
ax2.set_title('Error at Each Position')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# Plot 3: Box plot of errors
ax3 = axes[1, 0]
error_data = []
labels = []
colors = []
for name, _, color in ALGORITHMS:
    valid = [err for _, _, err in results[name] if err is not None]
    if valid:
        error_data.append(valid)
        labels.append(name)
        colors.append(color)
if error_data:
    bp = ax3.boxplot(error_data, labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
ax3.set_ylabel('Error (°)')
ax3.set_title('Error Distribution')
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: Mean error bar chart
ax4 = axes[1, 1]
means = []
names = []
cols = []
for name, _, color in ALGORITHMS:
    valid = [err for _, _, err in results[name] if err is not None]
    if valid:
        means.append(np.mean(valid))
        names.append(name)
        cols.append(color)
if means:
    bars = ax4.bar(names, means, color=cols, alpha=0.7)
    for bar, val in zip(bars, means):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, 
                f'{val:.1f}°', ha='center', fontsize=10)
ax4.set_ylabel('Mean Error (°)')
ax4.set_title('Algorithm Comparison')
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('doa_comparison_results.png', dpi=150)
print(f"\n    Plot saved: doa_comparison_results.png")
plt.show()

print("\nDone.")