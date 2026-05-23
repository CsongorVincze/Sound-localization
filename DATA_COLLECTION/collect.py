#!/usr/bin/env python3
"""
Autonomous DoA dataset collection — Pro Version (Restored OG Logic).

Combines the robust PyAudio recording with the ORIGINAL working motor control.
"""

import argparse
import csv
import json
import logging
import sys
import threading
import time
import wave
from datetime import datetime
from pathlib import Path

import numpy as np
import pyaudio
import serial
import serial.tools.list_ports
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
from respeaker_doa import ReSpeakerDoA
from source_pool import SourcePool
import download_sources

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_DIR     = Path(__file__).parent
SESSIONS_DIR = BASE_DIR / 'sessions'
SOURCES_DIR  = BASE_DIR / 'sources'

SAMPLE_RATE    = 16000
RECORD_SECS    = 3
WARMUP_SECS    = 0.1
CHUNK          = 1024
BAUD           = 9600
SERIAL_TIMEOUT = 15
MAX_RETRIES    = 5
BG_INTERVAL    = 600
MOTOR_COOL_SECS = 1

# ReSpeaker v2 mapping
CH_SLICE_START = 1
CH_SLICE_END   = 5

METADATA_FIELDS = [
    'clip_type', 'sound_type', 'sweep', 'position', 'doa_degrees',
    'recording', 'source', 'chip_doa_raw', 'chip_doa_corrected',
    'sample_rate', 'num_channels', 'duration_s',
]

# ---------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------

def _setup_logging(log_file: Path):
    fmt = '%(asctime)s  %(levelname)-8s  %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )

log = logging.getLogger(__name__)

def generate_noise(noise_type: str, duration_s: float) -> Path:
    SOURCES_DIR.mkdir(exist_ok=True)
    path = SOURCES_DIR / f"noise_{noise_type}.wav"
    if path.exists(): return path
    
    n_samples = int(duration_s * 16000)
    if noise_type == 'white':
        noise = np.random.uniform(-1, 1, n_samples)
    elif noise_type == 'pink':
        uneven = np.random.randn(n_samples)
        X = np.fft.rfft(uneven)
        S = np.sqrt(np.arange(len(X)) + 1)
        noise = np.fft.irfft(X / S, n=n_samples)
    elif noise_type == 'brown':
        uneven = np.random.randn(n_samples)
        X = np.fft.rfft(uneven)
        S = np.arange(len(X)) + 1
        noise = np.fft.irfft(X / S, n=n_samples)
    elif noise_type == 'blue':
        uneven = np.random.randn(n_samples)
        X = np.fft.rfft(uneven)
        S = np.sqrt(np.arange(len(X)) + 1)
        noise = np.fft.irfft(X * S, n=n_samples)
    elif noise_type == 'hum':
        t = np.linspace(0, duration_s, n_samples)
        noise = (np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 100 * t))
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
    
    noise = (noise / np.max(np.abs(noise)) * 0.4).astype(np.float32)
    sf.write(str(path), noise, 16000, subtype='PCM_16')
    return path

def find_respeaker(pa):
    for i in range(pa.get_device_count()):
        try:
            dev = pa.get_device_info_by_index(i)
            name = dev['name'].lower()
            if ('respeaker' in name or 'seeed' in name) and dev['maxInputChannels'] > 0:
                return i
        except: continue
    for i in range(pa.get_device_count()):
        try:
            dev = pa.get_device_info_by_index(i)
            if 'microphone array' in dev['name'].lower() and dev['maxInputChannels'] >= 6:
                return i
        except: continue
    return None

def find_output_device(pa):
    try: return pa.get_default_output_device_info()['index']
    except: return 0

def find_arduino():
    for p in serial.tools.list_ports.comports():
        desc = p.description.lower()
        if any(k in desc for k in ('arduino', 'ch340', 'ch341', 'usb serial', 'com')):
            return p.device
    return None

def configure_respeaker(doa_reader: ReSpeakerDoA):
    if not doa_reader.available or doa_reader._tuning is None:
        log.warning("Tuning interface unavailable. Check drivers.")
        return
    try:
        t = doa_reader._tuning
        log.info("Resetting ReSpeaker DSP for RAW capture...")
        t.write('AGCONOFF', 0)
        t.write('STATNOISEONOFF', 0)
        t.write('NONSTATNOISEONOFF', 0)
        t.write('ECHOONOFF', 0)
        t.write('HPFONOFF', 0)
        t.write('AGCGAIN', 35.0) 
    except Exception as e: log.error(f"ReSpeaker config error: {e}")

# ---------------------------------------------------------------------------
# EXACT ORIGINAL WORKING ARDUINO CONTROLLER
# ---------------------------------------------------------------------------

class Arduino:
    def __init__(self, port: str, start_pos: int = 0):
        self.port = port
        self.pos  = start_pos
        self._ser = None
        self._connect()

    def _connect(self):
        for attempt in range(MAX_RETRIES):
            try:
                self._ser = serial.Serial(self.port, BAUD, timeout=SERIAL_TIMEOUT)
                time.sleep(2)
                self._ser.reset_input_buffer()
                resp = self._readline()
                if resp != "READY":
                    log.warning(f"Unexpected startup response: '{resp}'")
                log.info(f"Arduino connected on {self.port}")
                return
            except serial.SerialException as e:
                log.warning(f"Arduino connect attempt {attempt+1}/{MAX_RETRIES}: {e}")
                time.sleep(3)
        raise RuntimeError(f"Cannot connect to Arduino on {self.port}")

    def _readline(self) -> str:
        return self._ser.readline().decode(errors='replace').strip()

    def _cmd(self, text: str):
        self._ser.write(f"{text}\n".encode())

    def rotate(self) -> bool:
        for attempt in range(MAX_RETRIES):
            try:
                self._cmd("ROTATE")
                resp = self._readline()
                if resp == "READY":
                    self.pos = (self.pos + 1) % 72
                    return True
                log.warning(f"ROTATE unexpected response '{resp}'")
            except serial.SerialException:
                self._reconnect()
        return False

    def reset(self) -> bool:
        if self.pos == 0: return True
        for p in range(self.pos, 0, -1):
            steps = (round(p * 2048 / 72) - round((p - 1) * 2048 / 72))
            for attempt in range(MAX_RETRIES):
                try:
                    self._cmd(f"STEP -{steps}")
                    if self._readline() == "READY": break
                except serial.SerialException: self._reconnect()
        self._cmd("ZERO"); self._readline()
        self._cmd("DEENERGIZE"); self._readline()
        self.pos = 0
        return True

    def deenergize(self) -> bool:
        self._cmd("DEENERGIZE")
        return self._readline() == "READY"

    def zero(self) -> bool:
        self._cmd("ZERO")
        if self._readline() == "READY":
            self.pos = 0
            return True
        return False

    def step_degrees(self, degrees: float) -> bool:
        steps = round(degrees * 2048 / 360)
        if steps == 0: return True
        for attempt in range(MAX_RETRIES):
            try:
                self._cmd(f"STEP {steps}")
                if self._readline() == "READY": return True
            except serial.SerialException: self._reconnect()
        return False

    def _reconnect(self):
        try: self._ser.close()
        except: pass
        time.sleep(2)
        self._connect()

    def close(self):
        if self._ser: self._ser.close()

class DummyMotor:
    def rotate(self): return True
    def reset(self): pass
    def zero(self): pass
    def step_degrees(self, d): return True
    def close(self): pass

# ---------------------------------------------------------------------------
# Session & Recording
# ---------------------------------------------------------------------------

class SessionLog:
    def __init__(self, session_dir: Path):
        self._fh = open(session_dir / 'metadata.csv', 'a', newline='', encoding='utf-8')
        self._writer = csv.DictWriter(self._fh, fieldnames=METADATA_FIELDS)
        if (session_dir / 'metadata.csv').stat().st_size == 0: self._writer.writeheader()

    def write(self, row: dict):
        self._writer.writerow(row)
        self._fh.flush()

    def close(self): self._fh.close()

def record_raw(pa, in_idx, out_idx, src_path=None):
    total_dur = RECORD_SECS + WARMUP_SECS
    n_frames = int(total_dur * SAMPLE_RATE)
    out_stream = None
    src_data = b""
    if src_path:
        data, sr = sf.read(src_path, dtype='int16')
        if data.ndim > 1: data = data.mean(axis=1).astype('int16')
        if sr != 16000:
            n_out = int(len(data) * 16000 / sr)
            data = np.interp(np.linspace(0, len(data)-1, n_out), np.arange(len(data)), data).astype('int16')
        target_f = int(total_dur * 16000)
        data = np.pad(data, (0, target_f - len(data))) if len(data) < target_f else data[:target_f]
        src_data = np.column_stack([data, data]).tobytes()
        out_stream = pa.open(format=pyaudio.paInt16, channels=2, rate=16000, output=True, output_device_index=out_idx)

    actual_ch = pa.get_device_info_by_index(in_idx)['maxInputChannels']
    in_stream = pa.open(format=pyaudio.paInt16, channels=actual_ch, rate=16000, input=True, input_device_index=in_idx, frames_per_buffer=CHUNK)
    frames = []
    ptr = 0
    for _ in range(0, int(n_frames / CHUNK)):
        if out_stream:
            out_stream.write(src_data[ptr : ptr + CHUNK * 4])
            ptr += CHUNK * 4
        frames.append(in_stream.read(CHUNK, exception_on_overflow=False))
    in_stream.stop_stream(); in_stream.close()
    if out_stream: out_stream.stop_stream(); out_stream.close()
    full = np.frombuffer(b''.join(frames), dtype=np.int16).reshape(-1, actual_ch)
    start = int(WARMUP_SECS * SAMPLE_RATE)
    return full[start:, 1:5] if actual_ch >= 6 else full[start:, :4]

# ---------------------------------------------------------------------------
# ORIGINAL WORKING CALIBRATION LOOP
# ---------------------------------------------------------------------------

def calibrate(motor):
    print("\n" + "=" * 55)
    print("  CALIBRATION — jog motor to 0°")
    print("  Type degrees to rotate (e.g.  5   or  -10)")
    print("  Positive = clockwise, negative = counter-clockwise")
    print("  Press Enter with no input to accept current position as 0°")
    print("=" * 55)
    total = 0.0
    while True:
        raw = input(f"  [{total:+.1f}° from start]  degrees to jog (or Enter to accept): ").strip()
        if raw == '': break
        try:
            deg = float(raw)
            if motor.step_degrees(deg):
                total += deg
                print(f"  Rotated {deg:+.1f}°  (total offset from start: {total:+.1f}°)")
            else: print("  Motor error — try again.")
        except ValueError: print("  Not a number — try again.")
    motor.zero()
    print("[*] Calibration complete. Position 0° set.")

# ---------------------------------------------------------------------------
# Main Runner
# ---------------------------------------------------------------------------

def run_session(args):
    pa = pyaudio.PyAudio()
    sid = datetime.now().strftime('%Y%m%d_%H%M%S')
    sess_dir = SESSIONS_DIR / f"session_{sid}"
    sess_dir.mkdir(parents=True, exist_ok=True)
    _setup_logging(sess_dir / 'collection.log')

    print("[*] Connecting hardware...")
    in_idx = args.in_device if args.in_device is not None else find_respeaker(pa)
    out_idx = args.out_device if args.out_device is not None else find_output_device(pa)
    
    if in_idx is None:
        log.error("ReSpeaker not found.")
        for i in range(pa.get_device_count()): print(f"  {i}: {pa.get_device_info_by_index(i)['name']}")
        pa.terminate(); return

    port = args.port or find_arduino()
    motor = Arduino(port) if port else DummyMotor()
    doa = ReSpeakerDoA()
    configure_respeaker(doa)

    # 1. OG CALIBRATION
    calibrate(motor)

    # 2. SOURCE PREPARATION
    print("\n[*] Checking sound sources...")
    try:
        noise_files = [generate_noise(n, RECORD_SECS+1) for n in ['white', 'pink', 'brown', 'blue', 'hum']]
        env_files = list(SOURCES_DIR.glob("env_*.wav"))
        if len(env_files) < 10: 
            print("  - Downloading ESC-50 sounds...")
            download_sources.download_esc50(SOURCES_DIR, 250)
            env_files = list(SOURCES_DIR.glob("env_*.wav"))
        voice_files = list(SOURCES_DIR.glob("cv_*.wav")) + list(SOURCES_DIR.glob("ls_*.wav"))
        if len(voice_files) < 50:
            print("  - Downloading human voices...")
            download_sources.download_librispeech(SOURCES_DIR, 1000)
            voice_files = list(SOURCES_DIR.glob("cv_*.wav")) + list(SOURCES_DIR.glob("ls_*.wav"))
    except Exception as e: log.error(f"Source prep error: {e}")

    stages = {'industry': noise_files, 'environmental': env_files, 'voice': voice_files}
    stages = {k: v for k, v in stages.items() if v}
    total_clips = int(args.hours * 3600 / (RECORD_SECS + 0.5))
    sweeps_per_stage = max(1, (total_clips // len(stages)) // 72)
    print(f"[*] Layout: {len(stages)} stages, {sweeps_per_stage} sweeps each.")

    sl = SessionLog(sess_dir)
    last_bg, done = 0, 0
    try:
        for stage_name, file_list in stages.items():
            print(f"\n>>> STARTING STAGE: {stage_name} <<<")
            (sess_dir / f"recordings/{stage_name}").mkdir(parents=True, exist_ok=True)
            for sweep in range(1, sweeps_per_stage + 1):
                for pos in range(72):
                    try:
                        if pos > 0: motor.rotate()
                        doa_deg = pos * 5.0
                        if time.time() - last_bg > BG_INTERVAL:
                            rec_bg = record_raw(pa, in_idx, out_idx)
                            bg_p = f"recordings/background/bg_{done:05d}.wav"
                            (sess_dir / 'recordings/background').mkdir(exist_ok=True, parents=True)
                            sf.write(sess_dir / bg_p, rec_bg, 16000)
                            sl.write({'clip_type':'background', 'sound_type':'ambient', 'sweep':sweep, 'position':pos, 'doa_degrees':doa_deg, 'recording':bg_p})
                            last_bg = time.time()

                        src_path = file_list[done % len(file_list)]
                        log.info(f"  [{stage_name}] Pos: {doa_deg:.1f}° | Src: {Path(src_path).name}")
                        rec = record_raw(pa, in_idx, out_idx, src_path)
                        if rec is not None:
                            raw_v = doa.read_median(n=3)
                            rec_name = f"doa_{int(doa_deg):03d}_{done:05d}.wav"
                            rec_path = f"recordings/{stage_name}/{rec_name}"
                            sf.write(sess_dir / rec_path, rec, 16000)
                            sl.write({
                                'clip_type':'source', 'sound_type':stage_name, 'sweep':sweep, 'position':pos, 
                                'doa_degrees':doa_deg, 'recording':rec_path, 'source':str(src_path),
                                'chip_doa_raw': raw_v, 'chip_doa_corrected': (raw_v or 0),
                                'sample_rate': 16000, 'num_channels': 4, 'duration_s': RECORD_SECS
                            })
                        done += 1
                    except Exception as e: log.error(f"Clip error: {e}"); continue
                motor.reset()
                time.sleep(MOTOR_COOL_SECS)
    except KeyboardInterrupt: pass
    finally:
        sl.close(); motor.close(); pa.terminate()
        print(f"\n[*] Session complete! Data saved to {sess_dir}")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--hours', type=float, default=1.5)
    p.add_argument('--port', default=None)
    p.add_argument('--in-device', type=int, default=None)
    p.add_argument('--out-device', type=int, default=None)
    p.add_argument('--lang', default='en')
    run_session(p.parse_args())
