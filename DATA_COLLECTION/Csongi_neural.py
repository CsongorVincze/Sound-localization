import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


SAMPLE_RATE = 16000
ANGLE_STEP = 5
NUM_CLASSES = 72
MAX_TAU = 25
PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
DEFAULT_DIRS = [
    Path("sessions/session_20260521_005414/recordings/voice"),
    Path("sessions/session_20260517_161547/recordings"),
]

class DoAnet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(128, NUM_CLASSES),
        )
    def forward(self, x):
        return self.net(x)
    
    
def angle_from_name(path):
    match = re.match(r"doa_(\d{3})(?:_\d+)?\.wav$", path.name)
    return int(match.group(1)) if match else None

def recording_group(path, label):
    match = re.match(r"doa_\d{3}_(\d+)\.wav$", path.name)
    if match:
        return f"{path.parent}:{int(match.group(1)) - label}"
    return str(path.parent)

def find_recordings(roots):
    files = []
    for root in roots:
        files.extend(Path(root).rglob("doa_*.wav"))
    return [path for path in sorted(files) if angle_from_name(path) is not None]

def gcc_phat_lags(x, y):
    n = 2 ** np.ceil(np.log2(len(x) + len(y)))
    x_fft = np.fft.rfft(x, n=n)
    y_fft = np.fft.rfft(y, n=n)
    cross = x_fft * np.conj(y_fft)
    cross /= np.maximum(np.abs(cross), 1e-12)
    cc = np.fft.irfft(cross, n=n)
    return np.concatenate((cc[-MAX_TAU:], cc[:MAX_TAU + 1]))


    
