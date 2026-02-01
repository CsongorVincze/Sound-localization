"""
Arduino Calibration Master - DoA Measurement System
Controls a stepper motor and sound source to measure DoA algorithm accuracy.
Specifically targets ReSpeaker v2.0 hardware.
"""
import serial
import serial.tools.list_ports
import time
import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
from my_algos import get_gcc_phat_angle

# =============================================================================
# CONFIGURATION
# =============================================================================
COM_PORT = None        # Set to 'COM3' etc. or None for auto-detect
BAUD_RATE = 9600
SAMPLE_RATE = 16000
RECORDING_DURATION = 1.0
STEP_INCREMENT = 5     # degrees
TOTAL_STEPS = 36       # 180 degrees total

# =============================================================================
# SETUP
# =============================================================================

print("=" * 60)
print(" Arduino/ReSpeaker Calibration System")
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
        # Check for ReSpeaker signature
        name = dev['name'].lower()
        if 'respeaker' in name or 'uac1.0' in name or 'seeed' in name:
            respeaker_id = i
            print(f"    Found ReSpeaker: {dev['name']} (ID: {i})")
            break

if respeaker_id is None:
    print("ERROR: ReSpeaker not found! Please connect device via USB.")
    ard.close()
    exit(1)

# =============================================================================
# ALIGNMENT PHASE
# =============================================================================

print("\n" + "=" * 60)
print(" Zero Alignment (Manual Servo Control)")
print("=" * 60)
print("    Enter a servo angle (0-180) to move the motor.")
print("    The system will play a sound and show the measured DoA.")
print("    Find the servo angle providing ~0° DoA.")
print("    Type 'q' or 'done' to start calibration from that position.")

current_servo_angle = 0
alignment_offset = 0

try:
    # Reset first
    ard.write(b'R')
    time.sleep(1)
    ard.read_all()
    
    while True:
        user_input = input(f"\n    Enter Servo Angle (current {current_servo_angle}°): ").strip().lower()
        
        if user_input in ['q', 'done', 'exit']:
            print(f"    Alignment accepted. Starting scan from {current_servo_angle}°.")
            alignment_offset = current_servo_angle
            break
            
        try:
            target_angle = int(user_input)
            if target_angle < 0 or target_angle > 180:
                print("    ERROR: Angle must be 0-180")
                continue
                
            # Move Servo
            # We don't have a direct 'Move to X' command in the simple Arduino code
            # We only have 'M' (step) and 'R' (reset).
            # So we reset and step up to target.
            
            # actually, let's just use the relative steps if possible, or Reset and step.
            # Since our Arduino code is simple (only 'M' steps by 5 deg), 
            # we might need to modify Arduino code to go to absolute position OR
            # just be clever here.
            
            # Current Arduino Logic:
            # 'R' -> 0
            # 'M' -> current + 5
            
            # To go to arbitrary angle X:
            # Reset, then send 'M' (X / 5) times.
            
            print(f"    Moving to {target_angle}°...", end=" ", flush=True)
            ard.write(b'R')
             # Wait for reset
            while True:
                line = ard.readline().decode().strip()
                if line == "RESET_DONE":
                    break
            
            steps_needed = int(target_angle / STEP_INCREMENT)
            for _ in range(steps_needed):
                ard.write(b'M')
                # Wait for ready each step or just spam? 
                # Better to wait to be safe, though slow.
                while str(ard.readline().decode().strip()) != "READY": pass
            
            current_servo_angle = target_angle
            print("OK")
            
            # Record & Measure
            print("    Ping...", end=" ", flush=True)
            recording = sd.rec(
                int(1.0 * SAMPLE_RATE),
                samplerate=SAMPLE_RATE,
                channels=6,
                device=respeaker_id,
                dtype='int16'
            )
            time.sleep(0.1)
            ard.write(b'P')
            sd.wait()
            
            # Calculate
            raw_audio = recording[:, 1:5].astype(np.float64)
            est_angle = get_gcc_phat_angle(raw_audio, SAMPLE_RATE)
            
            err = est_angle if est_angle < 180 else est_angle - 360
            print(f" DoA: {est_angle:5.1f}° (Error: {err:5.1f}°)")
            
        except ValueError:
            print("    Invalid input. Enter a number.")

except KeyboardInterrupt:
    print("\n    Alignment cancelled.")
    exit(0)

# =============================================================================
# MEASUREMENT LOOP
# =============================================================================

print("\n" + "=" * 60)
print(" Starting Calibration Phase")
print("=" * 60)
print(f"    Start Offset: {alignment_offset}°")
print(f"    Total Steps:  {TOTAL_STEPS}")
print(f"    Increment:    {STEP_INCREMENT}°")

# We are already at 'alignment_offset'. 
# We will scan 180 degrees relative to this? 
# OR just scan from here until 180?
# Typically user wants to START at 0° DoA (Physical X) and go to 180° DoA.

# Let's assume we start here and do TOTAL_STEPS.
# But we must ensure we don't hit physical limit (180).
# If alignment_offset is large, we might hit 180 servo limit soon.

remaining_space = 180 - alignment_offset
max_steps_possible = int(remaining_space / STEP_INCREMENT)

if max_steps_possible < TOTAL_STEPS:
    print(f"    WARNING: Servo limit (180°) reached in {max_steps_possible} steps.")
    print(f"    Reducing run to {max_steps_possible} steps.")
    TOTAL_STEPS = max_steps_possible

current_angle = 0 # Relative measurement angle (Ground Truth 0)
results = []    

try:
    for step in range(TOTAL_STEPS):
        print(f"\n--- Step {step + 1}/{TOTAL_STEPS} (Rel: {current_angle}°, Phys: {current_servo_angle}°) ---")
        
        # 1. Move Motor
        print("    Moving...", end=" ", flush=True)
        ard.write(b'M')
        
        # Wait for "READY"
        while True:
            line = ard.readline().decode().strip()
            if line == "READY":
                print("OK")
                break
        
        # 2. Record & Trigger Sound
        print(f"    Recording...", end=" ", flush=True)
        
        # Start recording (non-blocking)
        recording = sd.rec(
            int(RECORDING_DURATION * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=6,
            device=respeaker_id,
            dtype='int16'
        )
        
        # Trigger sound shortly after recording starts
        time.sleep(0.1)
        ard.write(b'P')
        
        # Wait for recording
        sd.wait()
        print("OK")
        
        # 3. Process Data
        print("    Calculating...", end=" ", flush=True)
        
        # Extract mic channels (1-4)
        # ReSpeaker 6-ch mode: ch0=processed, ch1-4=raw mics, ch5=playback
        raw_audio = recording[:, 1:5].astype(np.float64)
        
        est_angle = get_gcc_phat_angle(raw_audio, SAMPLE_RATE)
        
        # Calculate error (shortest path on circle)
        error = abs(est_angle - current_angle)
        if error > 180:
            error = 360 - error
            
        print("OK")
        print(f"    TRUE: {current_angle:5.1f}° | EST: {est_angle:5.1f}° | ERR: {error:5.1f}°")
        
        results.append((current_angle, est_angle, error))
        
        # Prepare for next step
        current_angle += STEP_INCREMENT
        if current_angle >= 360:
            current_angle -= 360

except KeyboardInterrupt:
    print("\n\nStopped by user.")

finally:
    ard.close()

# =============================================================================
# RESULTS
# =============================================================================

if results:
    true_angles = [r[0] for r in results]
    est_angles = [r[1] for r in results]
    errors = [r[2] for r in results]
    
    mean_err = np.mean(errors)
    max_err = np.max(errors)
    
    print("\n" + "=" * 60)
    print(f" Results Summary")
    print("=" * 60)
    print(f"    Mean Error: {mean_err:.1f}°")
    print(f"    Max Error:  {max_err:.1f}°")
    
    # Simple plot
    plt.figure(figsize=(10, 6))
    
    # Upper plot: Tracking
    plt.subplot(2, 1, 1)
    plt.plot(true_angles, est_angles, 'bo-', label='Estimated')
    plt.plot(true_angles, true_angles, 'r--', label='Reference')
    plt.title('DoA Tracking Accuracy')
    plt.ylabel('Angle (°)')
    plt.legend()
    plt.grid(True)
    
    # Lower plot: Error
    plt.subplot(2, 1, 2)
    plt.bar(true_angles, errors, width=STEP_INCREMENT*0.8)
    plt.axhline(mean_err, color='r', linestyle='--', label=f'Mean: {mean_err:.1f}°')
    plt.title('Abs. Error')
    plt.xlabel('Physical Angle (°)')
    plt.ylabel('Error (°)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

print("\nDone.")