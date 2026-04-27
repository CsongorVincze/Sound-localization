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
# STARSS23 from Zenodo (4-channel tetrahedral mic array recordings)
# DOI: 10.5281/zenodo.7880637
DATA_DIR = os.path.join(os.path.dirname(__file__), "starss23_data")
MIC_DIR = os.path.join(DATA_DIR, "mic_dev")
META_DIR = os.path.join(DATA_DIR, "metadata_dev")

ZENODO_BASE = "https://zenodo.org/records/7880637/files"
MIC_ZIP_URL = f"{ZENODO_BASE}/mic_dev.zip"
META_ZIP_URL = f"{ZENODO_BASE}/metadata_dev.zip"

BATCH_SIZE = 8
NUM_EPOCHS = 5
LEARNING_RATE = 1e-4
TARGET_CHANNELS = 4  # Tetrahedral mic array = 4 channels (matches ReSpeaker)

# STFT parameters
N_FFT = 512
WIN_LENGTH = 512
HOP_LENGTH = 256
TARGET_FRAMES = 128  # ~2 sec segments at sr=24000, hop=256

# Training parameters
STEPS_PER_EPOCH = 50
SEGMENT_DURATION_SEC = 2.0  # Length of audio segments to train on


# --- 1. Define the Simple Modifiable CNN ---
class SoundLocalizationCNN(nn.Module):
    # in_channels is 8 because we pass Real and Imaginary parts of the 4 mics
    def __init__(self, in_channels=8):
        super(SoundLocalizationCNN, self).__init__()

        # 2D CNN operating on the Spectrogram
        # Input shape: (Batch, Channels, Freq_bins, Time_frames)
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))  # Pool to a fixed size before FC
        )

        self.regressor = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)  # Output: sin(theta) and cos(theta)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.regressor(x)
        return x


# --- 2. Download Dataset from Zenodo ---
def download_and_extract(url, dest_dir, name):
    """Downloads a zip file from Zenodo and extracts it."""
    os.makedirs(dest_dir, exist_ok=True)
    zip_path = os.path.join(dest_dir, f"{name}.zip")

    # Get expected size
    try:
        resp_head = requests.head(url, allow_redirects=True, timeout=10)
        expected_size = int(resp_head.headers.get('content-length', 0))
    except Exception:
        expected_size = 0

    download_needed = True
    if os.path.exists(zip_path):
        actual_size = os.path.getsize(zip_path)
        if expected_size > 0 and actual_size == expected_size:
            print(f"  {name}.zip already exists and is complete, skipping download.")
            download_needed = False
        else:
            print(f"  {name}.zip is incomplete (got {actual_size}B / {expected_size}B). Re-downloading...")

    if download_needed:
        print(f"  Downloading {name} from Zenodo...")
        resp = requests.get(url, stream=True)
        resp.raise_for_status()
        total = int(resp.headers.get('content-length', 0))
        with open(zip_path, 'wb') as f, tqdm(total=total, unit='B', unit_scale=True, desc=name) as pbar:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

    # Check if already extracted
    extracted_marker = os.path.join(dest_dir, f".{name}_extracted")
    if not os.path.exists(extracted_marker):
        print(f"  Extracting {name}...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(dest_dir)
            # Create marker file
            with open(extracted_marker, 'w') as f:
                f.write("done")
        except zipfile.BadZipFile:
            print(f"  ERROR: {name}.zip is corrupted! Deleting so it can redownload next time.")
            os.remove(zip_path)
            raise
    else:
        print(f"  {name} already extracted.")


def ensure_dataset():
    """Downloads STARSS23 mic + metadata if not present."""
    print("Checking dataset...")
    download_and_extract(MIC_ZIP_URL, DATA_DIR, "mic_dev")
    download_and_extract(META_ZIP_URL, DATA_DIR, "metadata_dev")
    print("Dataset ready!\n")


# --- 3. Parse STARSS23 Metadata ---
def parse_metadata(meta_dir):
    """
    Parse STARSS23 CSV metadata files.
    Each CSV row: [frame, class_id, track_id, azimuth, elevation]
    Returns a dict: {filename_stem: [(frame, azimuth, elevation), ...]}
    """
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
    """
    Build a list of (audio_path, start_sample, end_sample, azimuth) tuples.
    We create fixed-length segments from each audio file, using the average
    azimuth of events within that segment as the label.
    """
    samples = []
    segment_samples = int(SEGMENT_DURATION_SEC * sr)

    audio_files = glob.glob(os.path.join(mic_dir, "**", "*.wav"), recursive=True)
    for audio_path in audio_files:
        stem = os.path.splitext(os.path.basename(audio_path))[0]
        if stem not in labels:
            continue

        # Get file info without loading
        info = sf.info(audio_path)
        total_samples = info.frames
        file_sr = info.samplerate

        events = labels[stem]
        # STARSS23 metadata uses 10Hz frame rate (100ms per frame)
        frame_rate = 10  # frames per second

        # Create segments
        for seg_start in range(0, total_samples - segment_samples, segment_samples):
            seg_end = seg_start + segment_samples

            # Find events within this segment
            seg_start_frame = int(seg_start / file_sr * frame_rate)
            seg_end_frame = int(seg_end / file_sr * frame_rate)

            seg_events = [e for e in events if seg_start_frame <= e[0] < seg_end_frame]

            if seg_events:
                # Average azimuth of active events in this segment
                avg_azimuth = np.mean([e[1] for e in seg_events])
                samples.append((audio_path, seg_start, segment_samples, avg_azimuth))

    return samples


# --- 4. Batch Preparation ---
def prepare_batch(batch_samples, device, spectrogram_transform):
    """Load audio segments, compute spectrograms, return batch tensors."""
    input_tensors = []
    labels = []

    for (audio_path, frame_offset, num_frames, azimuth) in batch_samples:
        try:
            # Use soundfile instead of torchaudio to avoid TorchCodec errors
            waveform_np, sr = sf.read(
                audio_path, start=frame_offset, frames=num_frames, dtype='float32', always_2d=True
            )
            # soundfile returns (time, channels), so we transpose to (channels, time)
            waveform = torch.tensor(waveform_np.T)
        except Exception as e:
            print(f"  [WARN] Skipping {audio_path}: {e}")
            continue

        # waveform shape: (channels, time)
        # STARSS23 mic array has exactly 4 channels
        if waveform.size(0) > TARGET_CHANNELS:
            waveform = waveform[:TARGET_CHANNELS, :]
        elif waveform.size(0) < TARGET_CHANNELS:
            repeat_factor = (TARGET_CHANNELS // waveform.size(0)) + 1
            waveform = waveform.repeat(repeat_factor, 1)[:TARGET_CHANNELS, :]

        # Compute Complex Spectrogram (power=None) -> (Channels, Freq, Time)
        # Without phase information, sound localization is nearly impossible!
        waveform = waveform.to(device)
        spec = spectrogram_transform(waveform)

        # Convert Complex to Real and Imaginary channels -> shape: (Channels, 2, Freq, Time)
        spec = torch.view_as_real(spec).permute(0, 3, 1, 2)
        # Flatten the Channels and Real/Imag dimensions -> shape: (Channels*2, Freq, Time)
        spec = spec.reshape(-1, spec.size(2), spec.size(3))

        # Truncate or pad to fixed time length
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


# --- 5. Training Loop ---
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Step 1: Download dataset
    ensure_dataset()

    # Step 2: Parse labels
    print("Parsing metadata...")
    labels = parse_metadata(META_DIR)
    print(f"  Found labels for {len(labels)} audio files.")

    # Step 3: Build training samples
    print("Building training segments...")
    samples = build_training_samples(MIC_DIR, labels)
    print(f"  Created {len(samples)} training segments.\n")

    if len(samples) == 0:
        print("ERROR: No training samples found. Check that mic_dev/ and metadata_dev/ have matching files.")
        # Debug: list what we have
        mic_files = glob.glob(os.path.join(MIC_DIR, "**", "*.wav"), recursive=True)
        meta_files = glob.glob(os.path.join(META_DIR, "**", "*.csv"), recursive=True)
        print(f"  mic_dev WAV files: {len(mic_files)}")
        print(f"  metadata_dev CSV files: {len(meta_files)}")
        if mic_files:
            print(f"  Example WAV: {mic_files[0]}")
        if meta_files:
            print(f"  Example CSV: {meta_files[0]}")
        if labels:
            print(f"  Example label key: {list(labels.keys())[0]}")
        return

    # Step 4: Setup model
    # power=None returns a complex tensor, preserving crucial phase differences!
    spectrogram_transform = torchaudio.transforms.Spectrogram(
        n_fft=N_FFT, win_length=WIN_LENGTH, hop_length=HOP_LENGTH, power=None
    ).to(device)

    # Initialize CNN model (in_channels=8 because 4 mics * 2 (Real/Imag))
    model = SoundLocalizationCNN(in_channels=8).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Step 5: Train
    print("Starting training on STARSS23 mic array data...")
    print(f"  Epochs: {NUM_EPOCHS}, Steps/epoch: {min(STEPS_PER_EPOCH, len(samples)//BATCH_SIZE)}")
    print(f"  Batch size: {BATCH_SIZE}, Channels: {TARGET_CHANNELS}\n")

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        batch_count = 0

        # Shuffle samples each epoch
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
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f} (Sin/Cos MSE)\n")

    # Save the final model
    torch.save(model.state_dict(), "sound_localization_cnn.pth")
    print("Training finished. Model saved to 'sound_localization_cnn.pth'.")


if __name__ == "__main__":
    train()
