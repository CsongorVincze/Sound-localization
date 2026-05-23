"""
28BYJ-48 stepper round-trip test.

Rotates forward N positions (default: full 72 = 360°) one step at a time,
then steps backward one increment at a time using the same logic as collect.py.
Watch the motor and verify it returns exactly to the marked start position.

Usage:
  python test_motor.py              # full 360° round-trip
  python test_motor.py --positions 36   # half circle
  python test_motor.py --port COM3
"""

import argparse
import time
import serial
import serial.tools.list_ports

STEPS_PER_REV  = 2048
N_POSITIONS    = 72
STEP_DEG       = 5
BAUD           = 9600
SERIAL_TIMEOUT = 15


def find_arduino():
    for p in serial.tools.list_ports.comports():
        if any(k in p.description.lower()
               for k in ('arduino', 'ch340', 'ch341', 'usb serial')):
            return p.device
    return None


def steps_at(pos):
    return round(pos * STEPS_PER_REV / N_POSITIONS)


def send(ser, text):
    ser.write(f"{text}\n".encode())
    return ser.readline().decode(errors='replace').strip()


def connect(port):
    ser = serial.Serial(port, BAUD, timeout=SERIAL_TIMEOUT)
    time.sleep(2)
    ser.reset_input_buffer()
    startup = ser.readline().decode(errors='replace').strip()
    print(f"Arduino: '{startup}'")
    return ser


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--port',      default=None)
    parser.add_argument('--positions', type=int, default=N_POSITIONS)
    args = parser.parse_args()

    port = args.port or find_arduino()
    if not port:
        print("Arduino not found. Use --port COMx.")
        return

    ser = connect(port)
    n = min(args.positions, N_POSITIONS)

    # ------------------------------------------------------------------ forward
    print(f"\n--- FORWARD: {n} positions ({n * STEP_DEG}°) ---")
    input("Mark the motor start position, then press Enter.")

    t0 = time.time()
    for pos in range(1, n + 1):
        resp = send(ser, "ROTATE")
        print(f"  → pos {pos:2d}  ({pos * STEP_DEG:3d}°)  {resp}")
    print(f"Forward done in {time.time()-t0:.1f}s")

    input("\nCheck end position, then press Enter to step backward.")

    # ----------------------------------------------------------------- backward
    print(f"\n--- BACKWARD: {n} positions ---")
    t0 = time.time()
    for p in range(n, 0, -1):
        steps = steps_at(p) - steps_at(p - 1)
        resp  = send(ser, f"STEP -{steps}")
        print(f"  ← pos {p:2d}  STEP -{steps}  {resp}")

    send(ser, "ZERO")
    send(ser, "DEENERGIZE")
    print(f"Backward done in {time.time()-t0:.1f}s")
    print("\nDid the motor return exactly to the marked position?")
    ser.close()


if __name__ == '__main__':
    main()
