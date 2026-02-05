"""
SNN Data Acquisition
Collects raw audio data for Spiking Neural Network training.
Sequence:
1. Move to angle
2. Start recording (1s)
3. Trigger sharp sound
4. Save raw audio
"""
import serial
import serial.tools.list_ports
import time
import sounddevice as sd
import numpy as np
import pickle
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================
COM_PORT = None
BAUD_RATE = 9600
SAMPLE_RATE = 16000
RECORDING_DURATION = 1.0  # seconds

# Rotation parameters
STEP_INCREMENT = 3     # degrees (Modified from 5)
TOTAL_STEPS = 60       # 180 degrees / 3 = 60 steps
START_ANGLE = 45       # Default starting angle
REPETITIONS = 3        # Number of samples per angle

OUTPUT_FILE = 'measure_1/snn_training_data.pkl'

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def send_goto(ser, angle):
    """Send GOTO command to move servo smoothly to target angle."""
    cmd = f"G{angle:03d}"
    ser.write(cmd.encode())
    # Wait for READY
    while True:
        try:
            line = ser.readline().decode().strip()
            if line == "READY":
                return True
            if line == "ERROR":
                print(f"Error moving to {angle}")
                return False
        except Exception as e:
            print(f"Serial error: {e}")
            return False

def send_play(ser):
    """Send PLAY command to trigger sound (Sharp sound)."""
    ser.write(b'P')

def get_respeaker_id():
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if dev['max_input_channels'] >= 4:
            name = dev['name'].lower()
            if 'respeaker' in name or 'uac1.0' in name or 'seeed' in name:
                return i
    return None

def run_calibration(ard):
    """Interactive calibration to set the 0-degree starting position."""
    print("\n" + "=" * 60)
    print(" Calibration Mode")
    print("=" * 60)
    print("    Enter an angle (0-180) to move servo.")
    print("    Type 'done' or press Enter to confirm this as the 0° start position.")
    
    current_angle = START_ANGLE
    
    # Move to default first
    print(f"    Moving to default {current_angle}°...", end=" ", flush=True)
    send_goto(ard, current_angle)
    print("Done")
    
    while True:
        try:
            user_input = input(f"\n    Servo [{current_angle}°] > ").strip().lower()
            
            if user_input in ['done', 'd', '']:
                print(f"    Confirmed start position: {current_angle}° (mapped to 0°)")
                return current_angle
            
            target = int(user_input)
            if 0 <= target <= 180:
                print(f"    Moving to {target}°...", end=" ", flush=True)
                send_goto(ard, target)
                current_angle = target
                print("Done")
            else:
                print("    Angle must be 0-180")
                
        except ValueError:
            print("    Invalid input. Enter number or 'done'.")
    return current_angle

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print(" SNN Data Acquisition")
    print("=" * 60)

    # 1. Connect to Arduino
    print("\n[1] Detecting Arduino...")
    ports = list(serial.tools.list_ports.comports())
    global COM_PORT
    if COM_PORT is None:
        for p in ports:
            if any(x in p.description.lower() for x in ['arduino', 'ch340', 'usb serial']):
                COM_PORT = p.device
                break
        if COM_PORT is None and ports:
            COM_PORT = ports[0].device

    if not COM_PORT:
        print("ERROR: No COM ports found!")
        return

    try:
        ard = serial.Serial(COM_PORT, BAUD_RATE, timeout=2)
        time.sleep(2)
        ard.read_all()
        print(f"    Connected to Arduino on {COM_PORT}")
    except Exception as e:
        print(f"ERROR: Could not open {COM_PORT}: {e}")
        return

    # 2. Connect to ReSpeaker
    print("\n[2] Detecting ReSpeaker...")
    respeaker_id = get_respeaker_id()
    if respeaker_id is None:
        print("ERROR: ReSpeaker not found!")
        ard.close()
        return
    print(f"    Using device ID: {respeaker_id}")

    # 3. Calibration
    start_offset = run_calibration(ard)

    # 4. Data Collection Loop
    print("\n[4] Starting Data Collection")
    
    # Calculate max steps based on start position
    remaining_space = 180 - start_offset
    max_steps_possible = int(remaining_space / STEP_INCREMENT)
    
    steps_to_run = min(TOTAL_STEPS, max_steps_possible)
    
    # +1 to include the end step
    total_steps_actual = steps_to_run 
    
    print(f"    Start Position: {start_offset}° (Label: 0°)")
    print(f"    Loop: {total_steps_actual} steps x {STEP_INCREMENT}°")
    print(f"    End Position: {start_offset + total_steps_actual * STEP_INCREMENT}°")
    print(f"    Repetitions per angle: {REPETITIONS}")
    
    if total_steps_actual < TOTAL_STEPS:
        print(f"    WARNING: Reduced steps from {TOTAL_STEPS} to {total_steps_actual} due to servo limit.")
    
    collected_data = [] # List of dicts

    try:
        # Move to start first
        print(f"    Ensuring at start position {start_offset}°...")
        send_goto(ard, start_offset)
        time.sleep(1)

        for step in range(total_steps_actual + 1):
            true_angle = step * STEP_INCREMENT
            servo_angle = start_offset + true_angle

            print(f"\n--- Step {step}/{total_steps_actual} | Label: {true_angle}° | Servo: {servo_angle}° ---")

            if step > 0:
                print(f"    Rotating...", end=" ", flush=True)
                send_goto(ard, servo_angle)
                print("Done")
            
            # Allow vibration to settle
            time.sleep(0.2)

            # Repetitions
            for rep in range(REPETITIONS):
                print(f"    Recording Rep {rep+1}/{REPETITIONS}...", end=" ", flush=True)
                
                # Start recording (non-blocking)
                recording = sd.rec(
                    int(RECORDING_DURATION * SAMPLE_RATE),
                    samplerate=SAMPLE_RATE,
                    channels=6,
                    device=respeaker_id,
                    dtype='int16'
                )
                
                # Short delay to ensure recording started
                time.sleep(0.1)
                
                # Trigger sharp sound
                send_play(ard)
                
                # Wait for recording to finish
                sd.wait()
                
                # Extract mic channels
                raw_audio = recording[:, 1:5]
                
                data_point = {
                    'angle': true_angle,      # SNN label (0, 3, 6...)
                    'servo_angle': servo_angle, # Absolute hardware position
                    'audio': raw_audio
                }
                collected_data.append(data_point)
                
                print("Done")
                
                # Small delay between repetitions
                time.sleep(0.5)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    finally:
        ard.close()

    # 5. Save Data
    if collected_data:
        print(f"\n[5] Saving {len(collected_data)} samples to {OUTPUT_FILE}...")
        try:
            with open(OUTPUT_FILE, 'wb') as f:
                pickle.dump(collected_data, f)
            print("    Success!")
            print(f"    Data format: List of dicts {{'angle', 'servo_angle', 'audio'}}")
        except Exception as e:
            print(f"    Error saving file: {e}")
    else:
        print("\nNo data collected.")

if __name__ == "__main__":
    main()
