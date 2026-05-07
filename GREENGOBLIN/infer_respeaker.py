import torch
import torch.nn as nn
import torchaudio
import numpy as np
import sounddevice as sd
import sys
import math
import os
import time

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "joint_sep_loc.pth")
SAMPLE_RATE = 16000
TARGET_CHANNELS = 4
N_FFT = 512
WIN_LENGTH = 512
HOP_LENGTH = 256
TARGET_FRAMES = 128  # Network expects exactly 128 frames
BUFFER_SIZE = HOP_LENGTH * (TARGET_FRAMES - 1)  # Roughly 2 seconds of audio at 16kHz

device = torch.device("cpu") # Run on CPU for edge device/laptop inference

# --- Model Definition ---
# (This must exactly match the architecture we trained)
class JointSepLocNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(8, 32, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(32), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.enc3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU())
        self.dec1 = nn.Sequential(nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU())
        self.mask_out = nn.Sequential(nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1), nn.Sigmoid())
        self.loc_pool = nn.AdaptiveMaxPool2d((1, None))
        self.rnn = nn.GRU(128, 64, num_layers=2, batch_first=True, bidirectional=True)
        self.loc_out = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 4))

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        d1 = self.dec1(e3)
        if d1.size() != e2.size(): d1 = torch.nn.functional.interpolate(d1, size=e2.shape[2:])
        d2 = self.dec2(d1 + e2)
        if d2.size() != e1.size(): d2 = torch.nn.functional.interpolate(d2, size=e1.shape[2:])
        masks = self.mask_out(d2 + e1)
        loc = self.loc_pool(e3).squeeze(2).permute(0, 2, 1)
        loc, _ = self.rnn(loc)
        doas = self.loc_out(loc[:, -1, :])
        return masks, doas

# --- Real-Time Inference ---
def main():
    print("Loading Joint Separation & Localization model...")
    model = JointSepLocNet()
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        print("Model loaded successfully!")
    except FileNotFoundError:
        print(f"Error: Could not find {MODEL_PATH}. Make sure it is in the same directory.")
        sys.exit(1)

    complex_spec_transform = torchaudio.transforms.Spectrogram(
        n_fft=N_FFT, win_length=WIN_LENGTH, hop_length=HOP_LENGTH, power=None
    ).to(device)

    print("\n[!] Starting live audio stream from ReSpeaker...")
    print("[!] Speak near the mic! Press Ctrl+C to stop.\n")

    def audio_callback(indata, frames, time_info, status):
        if status:
            pass # Ignore minor buffer underflows for real-time
            
        # indata shape is (frames, channels)
        # ReSpeaker v2 usually outputs 6 channels via USB. We ONLY want the first 4 (the raw mics).
        if indata.shape[1] >= 4:
            audio_buffer = indata[:, :4]
        else:
            audio_buffer = indata

        # Convert to tensor (Channels, Time)
        waveform = torch.tensor(audio_buffer.T, dtype=torch.float32)
        
        # Normalize
        waveform = waveform / (waveform.abs().max() + 1e-8)
        
        # STFT
        complex_spec = complex_spec_transform(waveform)
        
        # Convert to Real/Imag and shape to (8, F, T)
        spec = torch.view_as_real(complex_spec).permute(0, 3, 1, 2).reshape(-1, complex_spec.size(1), complex_spec.size(2))
        
        # Force exactly TARGET_FRAMES
        if spec.size(-1) > TARGET_FRAMES:
            spec = spec[..., :TARGET_FRAMES]
        elif spec.size(-1) < TARGET_FRAMES:
            spec = torch.nn.functional.pad(spec, (0, TARGET_FRAMES - spec.size(-1)))
            
        spec = spec.unsqueeze(0) # Add batch dimension -> (1, 8, F, T)

        # Run Neural Network
        with torch.no_grad():
            masks, doas = model(spec)
            
        # Extract angles
        # doas shape: (1, 4) -> [sin1, cos1, sin2, cos2]
        doas = doas[0].numpy()
        az1_rad = math.atan2(doas[0], doas[1])
        az2_rad = math.atan2(doas[2], doas[3])
        
        # Convert to 0-360 degrees
        az1_deg = (math.degrees(az1_rad) + 360) % 360
        az2_deg = (math.degrees(az2_rad) + 360) % 360
        
        print(f"\r🎤 Tracking Source 1: {az1_deg:05.1f}°   |   🎤 Tracking Source 2: {az2_deg:05.1f}°     ", end="", flush=True)

    # Find the ReSpeaker device automatically
    respeaker_idx = None
    for i, dev in enumerate(sd.query_devices()):
        if 'respeaker' in dev['name'].lower() and dev['max_input_channels'] > 0:
            respeaker_idx = i
            break

    if respeaker_idx is not None:
        print(f"\n[!] Found ReSpeaker device at index {respeaker_idx}: {sd.query_devices()[respeaker_idx]['name']}")
    else:
        print("\n[!] Could not find 'ReSpeaker' device automatically. Falling back to default.")
        print("If you get a channel error, check the list below to find your mic's device index:")
        print(sd.query_devices())

    try:
        # We try to request 6 channels (Standard ReSpeaker USB profile)
        with sd.InputStream(device=respeaker_idx, samplerate=SAMPLE_RATE, channels=6, blocksize=BUFFER_SIZE, callback=audio_callback):
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopped.")
    except Exception as e:
        print(f"\n[X] Error opening audio stream: {e}")
        print("\nTIP: Make sure you are running this script on the LOCAL machine where the ReSpeaker is plugged into the USB port, NOT on the SSH server!")
        print("You can check your audio devices by running: python -m sounddevice")

if __name__ == "__main__":
    main()
