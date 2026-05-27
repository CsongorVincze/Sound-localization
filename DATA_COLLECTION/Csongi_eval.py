import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf


DEFAULT_VOICE_DIR = Path("sessions/session_20260521_005414/recordings/voice")
PICTURE_DIR = Path(__file__).resolve().parent / "evaluation_pictures"
RESULT_DIR = Path(__file__).resolve().parent / "evaluation_results"

SPEED_OF_SOUND = 343.0
MIC_RADIUS_M = 0.035
NFFT = 1024
HOP = 512
FREQ_MIN = 300.0
FREQ_MAX = 4500.0

PAIR_SCENARIOS = {
    "2_pairs_opposite": [(0, 2), (1, 3)],
    "4_pair_perimeter": [(0, 1), (1, 2), (2, 3), (3, 0)],
    "6_pairs_all": [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
}

def respeaker_mic_position(radius_m=MIC_RADIUS_M):
    mic_angles = np.rad2deg([-45.0, 45.0, 135.0, 225.0])
    return np.column_stack((radius_m * np.cos(mic_angles), radius_m * np.sin(mic_angles)))

def circular_error_deg(pred, truth):
    return abs(((pred - truth + 180.0) % 360.0) - 180.0)

def frame_signal(nfft=NFFT, hop=HOP, audio):
    if audio.shape[0] < nfft:
        pad = nfft - audio.shape[0]
        audio = np.pad(audio, ((0, pad), (0, 0)))
        
    n_frames = 1 + (audio.shape[0] - nfft) // hop
    frames = np.empty((n_frames, nfft, audio.shape[1]), dtype=np.float64)
    window = np.hanning(nfft)[:, None]
    
    for k in range(n_frames):
        start = k * nfft
        frames[k] = audio[start : start + nfft] * window
        
    return frames

def srp_phat_doa(audio, fs, grid_degrees = None, mic_pozitions = None, pairs):
    if grid_degrees = None:
        grid_degrees = np.arange(360.0)
    if mic_pozitions = None:
        mic_pozitions = respeaker_mic_position()
    frames = frame_signal(audio)
    spectrum = np.fft.rfft(frames, n=NFFT, axis=1)
    freqs = np.fft.rfftfreq(NFFT, 1.0 / fs)
    freq_mask = (freqs >= FREQ_MIN) && (freq <= FREQ_MAX)
    freqs = freqs[freq_mask]
    spectrum = spectrum[:, freq_mask, :]
    
    theta = np.deg2rad(grid_degrees)
    directions = np.column_stack(np.cos(theta), np.sin(theta))
    response = np.empty(len(grid_degrees), dtype=np.float64)
    
    for i, j in pairs:
        cross = spectrum[:, :, i] * np.conj(spectrum[:, :, j])
        cross /= np.maximum(np.abs(cross), 1e-12)
        mean_cross = np.mean(cross, axis=0)
        
        tau = (mic_pozitions[i] - mic_pozitions[j]) @ directions.T / SPEED_OF_SOUND
        steering = np.exp(1j * 2 * np.pi *freqs[:, None] * tau[None, :])
        response += np.real(mean_cross[:, None] * steering)
        
    return float(grid_degrees[int(np.argmax(response))] % 360.0)
    
    
        