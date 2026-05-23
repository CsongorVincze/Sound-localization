"""
One-time feature extraction: Segmented_Sound .wav files -> preprocessed/features.npy

Usage:
    python preprocess.py --data_dir /path/to/SoClas_database --out_dir ./preprocessed

Output files (in out_dir):
    features.npy    (N, 27, 618) float32 - normalized GCC-PHAT + MFCC features
    doa_targets.npy (N, 360)     float32 - Gaussian DoA targets (sigma=6 deg)
    class_labels.npy (N,)        int64   - sound class index 0..9
    doa_angles.npy   (N,)        int16   - actual DoA in degrees 5..360
"""

import argparse
import numpy as np
import soundfile as sf
import torch
import torchaudio
from pathlib import Path
from tqdm import tqdm

from features import (
    SAMPLE_RATE, T_MAX, FEAT_DIM,
    extract_file_features, make_doa_target
)


def collect_files(sound_dir):
    """Returns list of (wav_path, class_idx, angle_deg) sorted deterministically."""
    entries = []
    for class_idx in range(10):
        class_name = f'class{class_idx + 1:02d}'
        class_dir = sound_dir / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Expected directory not found: {class_dir}")
        for angle_dir in sorted(class_dir.iterdir()):
            if not angle_dir.is_dir():
                continue
            try:
                angle_deg = int(angle_dir.name.split('_')[-1])
            except ValueError:
                continue
            for wav_path in sorted(angle_dir.glob('*.wav')):
                entries.append((wav_path, class_idx, angle_deg))
    return entries


def load_audio(wav_path):
    """Load .wav via soundfile (no TorchCodec dependency), enforce 4 channels,
    resample to SAMPLE_RATE, normalize peak."""
    data, sr = sf.read(str(wav_path), always_2d=True)  # (T, channels), float64
    audio = torch.from_numpy(data.T.astype(np.float32))  # (channels, T)

    if audio.shape[0] > 4:
        audio = audio[:4]
    elif audio.shape[0] < 4:
        audio = audio.repeat(4, 1)[:4]

    if sr != SAMPLE_RATE:
        audio = torchaudio.functional.resample(audio, sr, SAMPLE_RATE)

    peak = audio.abs().max()
    if peak > 0:
        audio = audio / peak

    return audio  # (4, T)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True,
                        help='Root of SoClas_database (contains Segmented_Sound/)')
    parser.add_argument('--out_dir', default='./preprocessed',
                        help='Output directory for .npy feature files')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sound_dir = data_dir / 'Segmented_Sound'
    print(f"Scanning {sound_dir} ...")
    entries = collect_files(sound_dir)
    N = len(entries)
    print(f"Found {N} files.")

    features    = np.zeros((N, T_MAX, FEAT_DIM), dtype=np.float32)
    doa_targets = np.zeros((N, 360),             dtype=np.float32)
    class_labels = np.zeros(N,                    dtype=np.int64)
    doa_angles  = np.zeros(N,                     dtype=np.int16)

    errors = []
    for i, (wav_path, class_idx, angle_deg) in enumerate(tqdm(entries, unit='file')):
        try:
            audio = load_audio(wav_path)
            features[i]      = extract_file_features(audio)
            doa_targets[i]   = make_doa_target(angle_deg, sigma=6.0)
            class_labels[i]  = class_idx
            doa_angles[i]    = angle_deg
        except Exception as e:
            errors.append((str(wav_path), str(e)))

    if errors:
        error_rate = len(errors) / N
        print(f"\n{len(errors)}/{N} files failed ({error_rate*100:.1f}%):")
        for path, err in errors[:10]:
            print(f"  {path}: {err}")
        if error_rate > 0.01:
            raise RuntimeError(
                f"{len(errors)} files failed ({error_rate*100:.1f}%) - "
                f"refusing to save corrupt dataset. Fix the loader and re-run."
            )
        print("(<1% failure rate - continuing)")

    np.save(out_dir / 'features.npy',     features)
    np.save(out_dir / 'doa_targets.npy',  doa_targets)
    np.save(out_dir / 'class_labels.npy', class_labels)
    np.save(out_dir / 'doa_angles.npy',   doa_angles)

    print(f"\nSaved to {out_dir}/")
    print(f"  features.npy:     {features.shape}  "
          f"({features.nbytes / 1e9:.1f} GB)")
    print(f"  doa_targets.npy:  {doa_targets.shape}")
    print(f"  class_labels.npy: {class_labels.shape}")
    print(f"  doa_angles.npy:   {doa_angles.shape}")


if __name__ == '__main__':
    main()
