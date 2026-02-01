"""
Arduino Calibration Master - DoA Measurement System
Controls a stepper motor and sound source to measure DoA algorithm accuracy.
"""
import serial
import serial.tools.list_ports
import time
import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
from my_algos import get_gcc_phat_angle, get_srp_phat_angle

# =============================================================================
# CONFIGURATION
# =============================================================================
COM_PORT = None        # Set to 'COM3', etc. to force specific port, or None for auto-detect
BAUD_RATE = 9600
SAMPLE_RATE = 16000
RECORDING_DURATION = 1.0  # seconds
STEP_INCREMENT = 5        # degrees per step
TOTAL_STEPS = 36          # 36 steps * 5° = 180° total

# =============================================================================
# DEVICE DETECTION
# =============================================================================

print("=" * 60)
print(" Arduino DoA Calibration System")
print("=" * 60)

# --- Find Arduino ---
print("\n[1] Detecting COM ports...")
ports = list(serial.tools.list_ports.comports())
if not ports:
    print("ERROR: No COM ports found! Is Arduino connected?")
    exit(1)

print("Available COM ports:")
for p in ports:
    print(f"    {p.device}: {p.description}")

if COM_PORT is None:
    # Auto-detect Arduino
    for p in ports:
        desc = p.description.lower()
        if any(x in desc for x in ['arduino', 'ch340', 'ch341', 'usb serial', 'usb-serial']):
            COM_PORT = p.device
            print(f"\n    Auto-detected Arduino on {COM_PORT}")
            break
    
    if COM_PORT is None:
        COM_PORT = ports[0].device
        print(f"\n    No Arduino detected, using first port: {COM_PORT}")

# Connect to Arduino
print(f"\n[2] Connecting to Arduino on {COM_PORT}...")
try:
    ard = serial.Serial(COM_PORT, BAUD_RATE, timeout=2)
    time.sleep(2)  # Wait for Arduino to reset after connection
    print(f"    Connected successfully!")
except Exception as e:
    print(f"ERROR: Could not connect to {COM_PORT}")
    print(f"       {e}")
    print("\n    TIPS:")
    print("    - Close Arduino IDE Serial Monitor if open")
    print("    - Check if correct COM port is selected")
    print("    - Try unplugging and replugging the Arduino")
    exit(1)

# --- Find Audio Device ---
print("\n[3] Detecting audio devices...")
devices = sd.query_devices()
respeaker_id = None
respeaker_channels = 0

for i, dev in enumerate(devices):
    if dev['max_input_channels'] >= 4:
        name = dev['name'].lower()
        if 'respeaker' in name or 'uac1.0' in name or 'seeed' in name:
            respeaker_id = i
            respeaker_channels = dev['max_input_channels']
            print(f"    Found ReSpeaker: {dev['name']}")
            print(f"    Device ID: {i}, Channels: {respeaker_channels}")
            break

if respeaker_id is None:
    # List all input devices
    print("    WARNING: ReSpeaker not found!")
    print("\n    Available input devices:")
    for i, dev in enumerate(devices):
        if dev['max_input_channels'] > 0:
            print(f"      [{i}] {dev['name']} ({dev['max_input_channels']} ch)")
    
    print("\n    TIPS:")
    print("    - Make sure ReSpeaker is connected via USB")
    print("    - It should appear as 'ReSpeaker 4 Mic Array (UAC1.0)'")
    print("\n    Do you want to continue with default audio device? (y/n)")
    response = input("    > ").strip().lower()
    if response != 'y':
        ard.close()
        exit(1)
    
    # Use first device with at least 2 channels
    for i, dev in enumerate(devices):
        if dev['max_input_channels'] >= 2:
            respeaker_id = i
            respeaker_channels = min(dev['max_input_channels'], 6)
            print(f"    Using: {dev['name']} ({respeaker_channels} channels)")
            break

# =============================================================================
# CALIBRATION RUN
# =============================================================================

print("\n" + "=" * 60)
print(" Starting Calibration Run")
print("=" * 60)
print(f"    Steps: {TOTAL_STEPS}")
print(f"    Increment: {STEP_INCREMENT}° per step")
print(f"    Total rotation: {TOTAL_STEPS * STEP_INCREMENT}°")
print("\n    Press Ctrl+C to stop at any time\n")

current_angle = 0
results = []

try:
    for step in range(TOTAL_STEPS):
        print(f"\n--- Step {step + 1}/{TOTAL_STEPS} ---")
        
        # 1. Move motor
        print("    Moving motor...", end=" ", flush=True)
        ard.write(b'M')
        
        # 2. Wait for motor to settle
        timeout_count = 0
        while True:
            line = ard.readline().decode().strip()
            if line == "READY":
                print("OK")
                break
            if line:
                print(f"[Arduino: {line}]", end=" ", flush=True)
            timeout_count += 1
            if timeout_count > 50:  # 50 * 2s timeout = 100s max wait
                print("TIMEOUT!")
                break
        
        # Update ground truth
        current_angle += STEP_INCREMENT
        if current_angle >= 360:
            current_angle -= 360
        
        # 3. Start recording
        print(f"    Recording ({RECORDING_DURATION}s)...", end=" ", flush=True)
        try:
            recording = sd.rec(
                int(RECORDING_DURATION * SAMPLE_RATE),
                samplerate=SAMPLE_RATE,
                channels=respeaker_channels,
                device=respeaker_id,
                dtype='int16'
            )
        except Exception as e:
            print(f"ERROR: {e}")
            continue
        
        # 4. Trigger sound
        time.sleep(0.1)
        ard.write(b'P')
        
        # 5. Wait for recording to finish
        sd.wait()
        print("OK")
        
        # 6. Calculate DoA
        print("    Calculating DoA...", end=" ", flush=True)
        try:
            # Extract mic channels (assuming channels 1-4 are mics in 6-ch mode)
            if respeaker_channels >= 6:
                raw_audio = recording[:, 1:5].astype(np.float64)
            elif respeaker_channels >= 4:
                raw_audio = recording[:, 0:4].astype(np.float64)
            else:
                # For 2 channels, can't do proper DoA
                raw_audio = recording.astype(np.float64)
                print("WARNING: Need 4+ channels for accurate DoA")
            
            est_angle = get_gcc_phat_angle(raw_audio, SAMPLE_RATE)
            
            # Calculate error (handle wraparound)
            error = abs(est_angle - current_angle)
            if error > 180:
                error = 360 - error
            
            print(f"OK")
            print(f"    TRUE: {current_angle:6.1f}° | EST: {est_angle:6.1f}° | ERROR: {error:5.1f}°")
            
            results.append((current_angle, est_angle, error))
            
        except Exception as e:
            print(f"ERROR: {e}")
            results.append((current_angle, None, None))

except KeyboardInterrupt:
    print("\n\n    Calibration stopped by user.")

finally:
    ard.close()
    print("\n    Arduino connection closed.")

# =============================================================================
# RESULTS
# =============================================================================

print("\n" + "=" * 60)
print(" Results")
print("=" * 60)

if not results:
    print("    No data collected.")
    exit(0)

# Filter out failed measurements
valid_results = [(t, e, err) for t, e, err in results if e is not None]

if valid_results:
    true_angles = [r[0] for r in valid_results]
    est_angles = [r[1] for r in valid_results]
    errors = [r[2] for r in valid_results]
    
    print(f"\n    Measurements: {len(valid_results)}/{len(results)} successful")
    print(f"    Mean error: {np.mean(errors):.1f}°")
    print(f"    Max error: {np.max(errors):.1f}°")
    print(f"    Std dev: {np.std(errors):.1f}°")
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: True vs Estimated
    ax1.plot(true_angles, est_angles, 'bo-', markersize=6, label='Measured')
    ax1.plot([0, 360], [0, 360], 'r--', linewidth=2, label='Perfect')
    ax1.set_xlabel('True Angle (°)', fontsize=12)
    ax1.set_ylabel('Estimated Angle (°)', fontsize=12)
    ax1.set_title('DoA Algorithm Performance', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, max(true_angles) + 10)
    ax1.set_ylim(0, 360)
    
    # Plot 2: Error distribution
    ax2.bar(true_angles, errors, width=STEP_INCREMENT*0.8, color='steelblue', alpha=0.7)
    ax2.axhline(y=np.mean(errors), color='r', linestyle='--', label=f'Mean: {np.mean(errors):.1f}°')
    ax2.set_xlabel('True Angle (°)', fontsize=12)
    ax2.set_ylabel('Error (°)', fontsize=12)
    ax2.set_title('Error at Each Position', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('calibration_results.png', dpi=150)
    print("\n    Plot saved to: calibration_results.png")
    plt.show()
else:
    print("    No valid measurements to plot.")

print("\n    Done!")