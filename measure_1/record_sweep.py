"""
record_sweep.py — Multi-Sound Audio Recording Sweep

Records 3 seconds of 6-channel audio at each servo angle position.
You name the output folder up front, then can do MULTIPLE recording sweeps
(one per sound type) without re-calibrating or reconnecting.

Folder structure:
    recordings/<your_name>/
        white_noise/      ← one sweep
            metadata.npy
            angle_000_servo_045.npy
            ...
        bach_goldberg/     ← another sweep
            metadata.npy
            angle_000_servo_045.npy
            ...
"""
import serial
import serial.tools.list_ports
import time
import sounddevice as sd
import numpy as np
from pathlib import Path
from datetime import datetime
from my_algos import get_gcc_phat_angle

# =============================================================================
# CONFIGURATION
# =============================================================================
COM_PORT = None
BAUD_RATE = 9600
SAMPLE_RATE = 16000
RECORDING_DURATION = 3.0   # seconds per position
STEP_INCREMENT = 5         # degrees
TOTAL_STEPS = 36           # 180 degrees total
DEFAULT_START_ANGLE = 45
SETTLE_TIME = 0.5
MIN_AUDIO_RMS = 50

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def send_goto(ser, angle):
    cmd = f"G{angle:03d}"
    ser.write(cmd.encode())
    while True:
        line = ser.readline().decode().strip()
        if line == "READY":
            return True
        if line == "ERROR":
            return False

def send_reset(ser):
    ser.write(b'R')
    while True:
        line = ser.readline().decode().strip()
        if line == "RESET_DONE":
            return

def check_audio_level(device_id, sample_rate, duration=0.5, channels=6):
    rec = sd.rec(int(duration * sample_rate), samplerate=sample_rate,
                 channels=channels, device=device_id, dtype='int16')
    sd.wait()
    mic_audio = rec[:, 1:5].astype(np.float64)
    return np.sqrt(np.mean(mic_audio ** 2))

def run_sweep(ard, respeaker_id, alignment_offset, out_dir, total_steps):
    """Execute one full recording sweep and save to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save metadata
    meta = {
        'sample_rate': SAMPLE_RATE,
        'recording_duration': RECORDING_DURATION,
        'step_increment': STEP_INCREMENT,
        'total_steps': total_steps,
        'alignment_offset': alignment_offset,
        'channels': 6,
        'mic_channels': '1:5',
    }
    np.save(out_dir / "metadata.npy", meta)

    print(f"\n    Output: {out_dir}")
    print(f"    Steps: {total_steps} x {STEP_INCREMENT}°, {RECORDING_DURATION}s each")
    print()

    for step in range(total_steps):
        true_angle = step * STEP_INCREMENT
        servo_pos = alignment_offset + true_angle

        print(f"--- Step {step+1}/{total_steps} | True: {true_angle}° | Servo: {servo_pos}° ---")

        print("    Moving...", end=" ", flush=True)
        send_goto(ard, servo_pos)
        print("OK")

        if SETTLE_TIME > 0:
            time.sleep(SETTLE_TIME)

        print(f"    Recording {RECORDING_DURATION}s...", end=" ", flush=True)
        recording = sd.rec(
            int(RECORDING_DURATION * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=6,
            device=respeaker_id,
            dtype='int16'
        )
        sd.wait()
        print("OK")

        fname = f"angle_{true_angle:03d}_servo_{servo_pos:03d}.npy"
        np.save(out_dir / fname, recording)
        print(f"    Saved: {fname}")

    # Return servo to start for the next sweep
    print("\n    Returning to start position...", end=" ", flush=True)
    send_goto(ard, alignment_offset)
    if SETTLE_TIME > 0:
        time.sleep(SETTLE_TIME)
    print("OK")


# =============================================================================
# MAIN
# =============================================================================
print("=" * 60)
print(" Multi-Sound Audio Recording Sweep")
print("=" * 60)

# --- 0. Name the session folder FIRST ---
print("\n    You will be able to record multiple sweeps (one per sound type)")
print("    inside a single session folder.\n")
session_name = input("    Name this session folder: ").strip()
if not session_name:
    session_name = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

session_dir = Path(__file__).parent / "recordings" / session_name
session_dir.mkdir(parents=True, exist_ok=True)
print(f"    ✓ Session folder: {session_dir}")

# --- 1. Connect to Arduino ---
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
    print(f"    Connected to Arduino on {COM_PORT}")
except Exception as e:
    print(f"ERROR: Could not open {COM_PORT}: {e}")
    exit(1)

# --- 2. Connect to ReSpeaker ---
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

# --- 3. Calibration (USB port alignment — done once) ---
print("\n" + "=" * 60)
print(" Calibration — USB Port Alignment")
print("=" * 60)
print("    Move the servo until the speaker is directly in front")
print("    of the ReSpeaker's USB port. That direction = 0°.")
print("    Enter a servo angle (0-180) or 'done' when aligned.")

current_servo_angle = DEFAULT_START_ANGLE
print(f"\n    Moving to default ({DEFAULT_START_ANGLE}°)...", end=" ", flush=True)
send_goto(ard, DEFAULT_START_ANGLE)
print("OK")

try:
    while True:
        user_input = input(f"\n    Servo [{current_servo_angle}°] > ").strip().lower()
        if user_input in ['done', 'd', 'q', '']:
            break
        try:
            target = int(user_input)
            if target < 0 or target > 180:
                print("    Error: must be 0-180")
                continue
            print(f"    Moving to {target}°...", end=" ", flush=True)
            send_goto(ard, target)
            current_servo_angle = target
            print("OK")
            time.sleep(SETTLE_TIME)
            rec = sd.rec(int(1.0 * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                         channels=6, device=respeaker_id, dtype='int16')
            sd.wait()
            raw = rec[:, 1:5].astype(np.float64)
            a = get_gcc_phat_angle(raw, SAMPLE_RATE)
            print(f"    GCC-PHAT reading: {a:.1f}°")
        except ValueError:
            print("    Invalid input.")
except KeyboardInterrupt:
    print("\n    Cancelled.")
    ard.close()
    exit(0)

alignment_offset = current_servo_angle

remaining_space = 180 - alignment_offset
max_steps = int(remaining_space / STEP_INCREMENT)
if max_steps < TOTAL_STEPS:
    print(f"    WARNING: Limited to {max_steps} steps (servo limit)")
    TOTAL_STEPS = max_steps

print(f"\n    ✓ Calibrated at servo={alignment_offset}° (= 0° true angle)")
print(f"    Available steps: {TOTAL_STEPS} x {STEP_INCREMENT}°")

# --- 4. Multi-sweep loop ---
sweep_count = 0

try:
    while True:
        print("\n" + "=" * 60)
        print(" New Recording Sweep")
        print("=" * 60)
        print("    Start a new sweep for a different sound type.")
        print("    The speaker must be playing the new sound.")
        print("    Type 'quit' to finish.\n")

        sound_name = input("    Name this sound (e.g. 'white_noise', 'bach_goldberg'): ").strip()
        if sound_name.lower() in ['quit', 'q', 'exit']:
            break
        if not sound_name:
            sound_name = f"sound_{sweep_count + 1}"

        # Speaker check
        print("\n    Make sure the CORRECT sound is playing on the speaker.")
        input("    Press ENTER when ready... ")

        audio_rms = check_audio_level(respeaker_id, SAMPLE_RATE)
        if audio_rms < MIN_AUDIO_RMS:
            print(f"    WARNING: Audio level very low (RMS={audio_rms:.0f})")
            cont = input("    Continue anyway? (y/n): ").strip().lower()
            if cont not in ['y', 'yes']:
                continue
        else:
            print(f"    ✓ Speaker detected (RMS={audio_rms:.0f})")

        sweep_dir = session_dir / sound_name
        if sweep_dir.exists():
            print(f"    WARNING: '{sound_name}' already exists in this session!")
            overwrite = input("    Overwrite? (y/n): ").strip().lower()
            if overwrite not in ['y', 'yes']:
                continue

        print(f"\n{'='*60}")
        print(f" Recording: {sound_name}")
        print(f"{'='*60}")

        run_sweep(ard, respeaker_id, alignment_offset, sweep_dir, TOTAL_STEPS)
        sweep_count += 1

        print(f"\n    ✓ Sweep '{sound_name}' complete! ({sweep_count} sweep(s) done)")
        print("    You can now change the sound on the speaker for the next sweep.")

except KeyboardInterrupt:
    print("\n\nStopped by user.")

finally:
    ard.close()

print(f"\n{'='*60}")
print(f" Session complete!")
print(f" {sweep_count} sweep(s) saved to: {session_dir}")
print(f"{'='*60}")
print("\nRun evaluate_all.py on individual sweep folders to compare algorithms.")
print(f"  Example: python evaluate_all.py \"{session_dir / '<sound_name>'}\"")
