"""
Live keyword spotting + robot control on Raspberry Pi with ReSpeaker.

Usage:
    python infer.py
    python infer.py --model best_res15_speech.pth --threshold 0.70
"""

import argparse
import sys
import threading
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from dataset import CLASS_NAMES, LABEL_MAP, NUM_CLASSES, SAMPLE_RATE, CLIP_SAMPLES, wav_to_spec
from model import Res15

HOP_SAMPLES   = 8000
SILENCE_LABEL = LABEL_MAP['silence']
UNKNOWN_LABEL = LABEL_MAP['unknown']
DEBOUNCE_S    = 1.5

# ── Robot (soft-fail: works as monitor-only if hardware absent) ────────────── #
try:
    from robot import Robot
    _robot = Robot()
    print("Robot initialized.")
except Exception as e:
    _robot = None
    print(f"Robot unavailable ({e}) — monitor-only mode.")

# ── ReSpeaker hardware DoA (soft-fail) ────────────────────────────────────── #
sys.path.insert(0, str(Path(__file__).parent.parent / 'respeakeres_fileok'))
mic_tuning = None
try:
    import usb.core
    import usb.util
    from tuning import Tuning
    _dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
    if _dev:
        mic_tuning = Tuning(_dev)
        print("ReSpeaker hardware DoA initialized.")
    else:
        print("ReSpeaker USB device not found — hardware DoA disabled.")
except Exception:
    pass


def _dispatch(cmd: str, doa: float):
    """Run the robot action for cmd in a daemon thread so the audio loop never blocks."""
    if _robot is None:
        return
    if cmd == 'stop':
        _robot.stop()
        return
    actions = {
        'go':       lambda: _robot.go_towards(doa),
        'forward':  _robot.forward,
        'backward': _robot.backward,
        'left':     lambda: _robot.turn_left(90),
        'right':    lambda: _robot.turn_right(90),
    }
    action = actions.get(cmd)
    if action:
        threading.Thread(target=action, daemon=True).start()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',     default='best_res15_speech.pth')
    parser.add_argument('--threshold', type=float, default=0.70)
    args = parser.parse_args()

    device = torch.device('cpu')
    model  = Res15(num_classes=NUM_CLASSES)
    sd_    = torch.load(args.model, map_location=device, weights_only=True)
    sd_    = {k[7:] if k.startswith('module.') else k: v for k, v in sd_.items()}
    model.load_state_dict(sd_)
    model.eval()
    print(f"Model loaded.  Threshold: {args.threshold:.0%}\n")

    ring          = np.zeros(CLIP_SAMPLES, dtype=np.float32)
    last_cmd      = None
    last_cmd_time = 0.0

    def callback(indata, frames, time_info, status):
        nonlocal last_cmd, last_cmd_time

        mono = indata[:, :4].mean(axis=1).astype(np.float32)
        ring[:] = np.roll(ring, -frames)
        ring[-frames:] = mono

        wav  = torch.from_numpy(ring.copy())
        spec = wav_to_spec(wav)
        with torch.no_grad():
            probs = torch.softmax(model(spec.unsqueeze(0))[0], dim=0)

        conf, pred = probs.max(dim=0)
        pred_idx = pred.item()
        conf_val = conf.item()
        cmd_name = CLASS_NAMES[pred_idx]

        doa = 0
        hw_str = ''
        if mic_tuning:
            try:
                doa = mic_tuning.direction
                hw_str = f"  DoA: {doa:03d}°"
            except Exception:
                pass

        is_cmd = pred_idx not in (SILENCE_LABEL, UNKNOWN_LABEL) and conf_val >= args.threshold

        if is_cmd:
            now   = time.time()
            fresh = cmd_name != last_cmd or (now - last_cmd_time) > DEBOUNCE_S
            if fresh:
                last_cmd      = cmd_name
                last_cmd_time = now
                _dispatch(cmd_name, doa)
            print(f"\r>>> {cmd_name.upper():<12} ({conf_val:.1%}){hw_str}   ", end='', flush=True)
        else:
            print(f"\r    {cmd_name.upper():<12} ({conf_val:.1%}){hw_str}   ", end='', flush=True)

    print(f"Listening — ring: {CLIP_SAMPLES/SAMPLE_RATE:.1f}s  hop: {HOP_SAMPLES/SAMPLE_RATE:.1f}s  Ctrl+C to stop\n")
    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=6,
                            blocksize=HOP_SAMPLES, callback=callback):
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print('\nStopped.')
    finally:
        if _robot:
            _robot.shutdown()


if __name__ == '__main__':
    main()
