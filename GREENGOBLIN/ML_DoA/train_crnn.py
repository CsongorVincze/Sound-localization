import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import numpy as np
from tqdm import tqdm
import os
import zipfile
import requests
import csv
import glob
import soundfile as sf

# --- Configuration ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "starss23_data")
MIC_DIR = os.path.join(DATA_DIR, "mic_dev")
META_DIR = os.path.join(DATA_DIR, "metadata_dev")

BATCH_SIZE = 8
NUM_EPOCHS = 5
LEARNING_RATE = 1e-4
TARGET_CHANNELS = 4

# STFT parameters
N_FFT = 512
WIN_LENGTH = 512
HOP_LENGTH = 256
TARGET_FRAMES = 128  # ~2 sec segments at sr=24000, hop=256

STEPS_PER_EPOCH = 50
SEGMENT_DURATION_SEC = 2.0


# --- 1. Define the CRNN ---
# Convolutional Recurrent Neural Network (industry standard for spatial audio)
class SoundLocalizationCRNN(nn.Module):
    # in_channels is 8 because we pass Real and Imaginary parts of the 4 mics
    def __init__(self, in_channels=8):
        super(SoundLocalizationCRNN, self).__init__()

        # CNN block (extracts spatial and spectral features)
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)), # Pool freq and time

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # Pool frequency axis completely, keep time axis
            nn.AdaptiveMaxPool2d((1, None)) 
        )

        # RNN block (models temporal continuity of sound)
        # Input to GRU must be (batch, time, features)
        self.rnn = nn.GRU(input_size=128, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True)

        # Regressor
        self.regressor = nn.Sequential(
            nn.Linear(64 * 2, 64), # *2 because bidirectional
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)  # Output: sin(theta) and cos(theta)
        )

    def forward(self, x):
        # x shape: (Batch, Channels, Freq, Time)
        x = self.cnn(x)
        # x shape: (Batch, 128, 1, Time)
        x = x.squeeze(2) # Remove freq dim -> (Batch, 128, Time)
        x = x.permute(0, 2, 1) # Swap to (Batch, Time, Features) for GRU

        x, _ = self.rnn(x) # x shape: (Batch, Time, 128)
        
        # Take the output of the last time step
        x = x[:, -1, :] 
        
        x = self.regressor(x)
        return x


# --- 2. Helper Functions ---
def parse_metadata(meta_dir):
    labels = {}
    csv_files = glob.glob(os.path.join(meta_dir, "**", "*.csv"), recursive=True)

    for csv_path in csv_files:
        stem = os.path.splitext(os.path.basename(csv_path))[0]
        events = []
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 5:
                    frame = int(row[0])
                    azimuth = float(row[3])
                    elevation = float(row[4])
                    events.append((frame, azimuth, elevation))
        if events:
            labels[stem] = events
    return labels

def build_training_samples(mic_dir, labels, sr=24000):
    samples = []
    segment_samples = int(SEGMENT_DURATION_SEC * sr)
    audio_files = glob.glob(os.path.join(mic_dir, "**", "*.wav"), recursive=True)
    
    for audio_path in audio_files:
        stem = os.path.splitext(os.path.basename(audio_path))[0]
        if stem not in labels:
            continue

        info = sf.info(audio_path)
        total_samples = info.frames
        file_sr = info.samplerate
        events = labels[stem]
        frame_rate = 10 

        for seg_start in range(0, total_samples - segment_samples, segment_samples):
            seg_end = seg_start + segment_samples
            seg_start_frame = int(seg_start / file_sr * frame_rate)
            seg_end_frame = int(seg_end / file_sr * frame_rate)

            seg_events = [e for e in events if seg_start_frame <= e[0] < seg_end_frame]
            if seg_events:
                avg_azimuth = np.mean([e[1] for e in seg_events])
                samples.append((audio_path, seg_start, segment_samples, avg_azimuth))
    return samples

def prepare_batch(batch_samples, device, spectrogram_transform):
    input_tensors = []
    labels = []

    for (audio_path, frame_offset, num_frames, azimuth) in batch_samples:
        try:
            waveform_np, sr = sf.read(
                audio_path, start=frame_offset, frames=num_frames, dtype='float32', always_2d=True
            )
            waveform = torch.tensor(waveform_np.T)
        except Exception as e:
            continue

        if waveform.size(0) > TARGET_CHANNELS:
            waveform = waveform[:TARGET_CHANNELS, :]
        elif waveform.size(0) < TARGET_CHANNELS:
            repeat_factor = (TARGET_CHANNELS // waveform.size(0)) + 1
            waveform = waveform.repeat(repeat_factor, 1)[:TARGET_CHANNELS, :]

        waveform = waveform.to(device)
        
        # Compute Complex Spectrogram (power=None) -> (Channels, Freq, Time)
        # Without phase information, sound localization is nearly impossible!
        spec = spectrogram_transform(waveform)
        
        # Convert Complex to Real and Imaginary channels -> shape: (Channels, 2, Freq, Time)
        spec = torch.view_as_real(spec).permute(0, 3, 1, 2)
        # Flatten the Channels and Real/Imag dimensions -> shape: (Channels*2, Freq, Time)
        spec = spec.reshape(-1, spec.size(2), spec.size(3))

        if spec.size(2) > TARGET_FRAMES:
            spec = spec[:, :, :TARGET_FRAMES]
        elif spec.size(2) < TARGET_FRAMES:
            pad_amount = TARGET_FRAMES - spec.size(2)
            spec = torch.nn.functional.pad(spec, (0, pad_amount))

        input_tensors.append(spec)
        
        # Convert degrees to radians and store [sin(theta), cos(theta)]
        az_rad = np.radians(azimuth)
        labels.append([np.sin(az_rad), np.cos(az_rad)])

    if len(input_tensors) == 0:
        return None, None

    batch_x = torch.stack(input_tensors)
    batch_y = torch.tensor(labels, dtype=torch.float32, device=device)
    return batch_x, batch_y


# --- 3. Training Loop ---
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    print("Parsing metadata...")
    labels = parse_metadata(META_DIR)
    
    print("Building training segments...")
    samples = build_training_samples(MIC_DIR, labels)
    print(f"  Created {len(samples)} training segments.\n")

    if len(samples) == 0:
        print("ERROR: No training samples found.")
        return

    # power=None returns a complex tensor, preserving crucial phase differences!
    spectrogram_transform = torchaudio.transforms.Spectrogram(
        n_fft=N_FFT, win_length=WIN_LENGTH, hop_length=HOP_LENGTH, power=None
    ).to(device)

    # Initialize CRNN model (in_channels=8 because 4 mics * 2 (Real/Imag))
    model = SoundLocalizationCRNN(in_channels=8).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("Starting training with CRNN architecture...")
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        batch_count = 0
        np.random.shuffle(samples)

        steps = min(STEPS_PER_EPOCH, len(samples) // BATCH_SIZE)
        progress_bar = tqdm(range(steps), desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")

        for step in progress_bar:
            batch_start = step * BATCH_SIZE
            batch_samples = samples[batch_start:batch_start + BATCH_SIZE]
            inputs, targets = prepare_batch(batch_samples, device, spectrogram_transform)

            if inputs is None:
                continue

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batch_count += 1
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_loss = running_loss / max(1, batch_count)
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f} (Sin/Cos MSE)")

    torch.save(model.state_dict(), "sound_localization_crnn.pth")
    print("Training finished. Model saved to 'sound_localization_crnn.pth'.")

if __name__ == "__main__":
    train()
