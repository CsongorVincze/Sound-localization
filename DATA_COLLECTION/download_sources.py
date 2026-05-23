"""
Download source audio clips for DoA data collection.

Option A — Mozilla Common Voice (default):
  Requires accepting the license at:
  https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0
  Then log in:  huggingface-cli login

Option B — LibriSpeech test-clean (public domain, no login):
  python download_sources.py --dataset librispeech

Usage:
  pip install datasets huggingface_hub soundfile numpy
  python download_sources.py [--n 2000] [--output sources/] [--lang en]
"""

import argparse
import io
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


def _resample_linear(arr: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return arr
    n_out = int(len(arr) * dst_sr / src_sr)
    return np.interp(
        np.linspace(0, len(arr) - 1, n_out),
        np.arange(len(arr)),
        arr,
    ).astype(np.float32)


def _save_clip(arr: np.ndarray, sr: int, path: Path, target_sr: int = 16000,
               target_s: float = 3.0) -> bool:
    if arr.ndim > 1:
        arr = arr.mean(axis=1).astype(np.float32)
    arr = _resample_linear(arr.astype(np.float32), sr, target_sr)
    target_n = int(target_s * target_sr)
    if len(arr) < int(target_sr * 1.0):   # skip clips shorter than 1 s
        return False
    arr = arr[:target_n] if len(arr) >= target_n else np.pad(arr, (0, target_n - len(arr)))
    sf.write(str(path), arr, target_sr, subtype='PCM_16')
    return True


def _decode_audio(sample: dict) -> tuple:
    """
    Decode audio from a HuggingFace dataset sample without torchcodec.
    Works regardless of the datasets library version by using Audio(decode=False)
    and decoding the raw bytes manually with soundfile.
    Returns (numpy_array, sample_rate).
    """
    audio = sample['audio']
    raw = audio.get('bytes')
    if raw:
        arr, sr = sf.read(io.BytesIO(raw), dtype='float32')
    else:
        # Streaming datasets sometimes provide a path instead of bytes
        arr, sr = sf.read(audio['path'], dtype='float32')
    return np.array(arr, dtype=np.float32), sr


def download_common_voice(output_dir: Path, n: int, lang: str):
    try:
        from datasets import load_dataset, Audio
    except ImportError:
        print("Install:  pip install datasets huggingface_hub")
        sys.exit(1)

    print(f"Loading Mozilla Common Voice 17.0 ({lang}, streaming) …")
    try:
        ds = load_dataset(
            "mozilla-foundation/common_voice_17_0",
            lang,
            split="train",
            streaming=True,
        )
        # decode=False: bypass torchcodec, decode bytes ourselves with soundfile
        ds = ds.cast_column("audio", Audio(decode=False))
    except Exception as e:
        print(f"Failed to load Common Voice: {e}")
        print("Have you accepted the license and run `huggingface-cli login`?")
        sys.exit(1)

    offset = len(list(output_dir.glob('cv_*.wav')))
    saved = 0
    for sample in ds:
        if saved >= n:
            break
        try:
            arr, sr = _decode_audio(sample)
        except Exception as e:
            print(f"  Skipping clip: {e}")
            continue
        path = output_dir / f"cv_{offset + saved:04d}.wav"
        if _save_clip(arr, sr, path):
            saved += 1
            if saved % 20 == 0:
                print(f"  {saved}/{n}")

    print(f"Saved {saved} Common Voice clips → {output_dir}")


def download_librispeech(output_dir: Path, n: int):
    try:
        from datasets import load_dataset, Audio
    except ImportError:
        print("Install:  pip install datasets")
        sys.exit(1)

    print("Loading LibriSpeech test-clean (public domain) …")
    ds = load_dataset("librispeech_asr", "clean", split="test", streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))

    offset = len(list(output_dir.glob('ls_*.wav')))
    saved = 0
    for sample in ds:
        if saved >= n:
            break
        try:
            arr, sr = _decode_audio(sample)
        except Exception as e:
            print(f"  Skipping clip: {e}")
            continue
        path = output_dir / f"ls_{offset + saved:04d}.wav"
        if _save_clip(arr, sr, path):
            saved += 1
            if saved % 20 == 0:
                print(f"  {saved}/{n}")

    print(f"Saved {saved} LibriSpeech clips → {output_dir}")


def download_esc50(output_dir: Path, n: int):
    try:
        from datasets import load_dataset, Audio
    except ImportError:
        print("Install:  pip install datasets soundfile")
        sys.exit(1)

    print("Loading ESC-50 dataset (Full Variety) …")
    # Mapping a wide range of useful environmental noises
    target_classes = {
        0: 'dog', 1: 'rooster', 2: 'pig', 3: 'cow', 4: 'frog', 5: 'cat', 
        10: 'rain', 11: 'sea_waves', 12: 'crackling_fire', 13: 'crickets',
        20: 'crying_baby', 21: 'sneezing', 22: 'clapping', 25: 'footsteps',
        30: 'door_wood_creaks', 31: 'mouse_click', 32: 'keyboard_typing',
        35: 'washing_machine', 36: 'vacuum_cleaner', 38: 'clock_tick',
        40: 'helicopter', 41: 'chainsaw', 42: 'siren', 43: 'car_horn', 
        44: 'engine', 45: 'train', 46: 'church_bells', 47: 'airplane'
    }
    
    ds = load_dataset("ashraq/esc50", split="train", streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))

    saved = 0
    # Track counts per class to keep it balanced
    class_counts = {k: 0 for k in target_classes.keys()}
    max_per_class = max(1, n // len(target_classes)) + 1

    for sample in ds:
        target = sample.get('target', sample.get('label'))
        if target not in target_classes or class_counts[target] >= max_per_class:
            continue
            
        if saved >= n:
            break
            
        try:
            arr, sr = _decode_audio(sample)
        except Exception as e:
            print(f"  Skipping clip: {e}")
            continue
            
        cat_name = target_classes[target]
        path = output_dir / f"env_{cat_name}_{saved:04d}.wav"
        if _save_clip(arr, sr, path):
            saved += 1
            class_counts[target] += 1
            if saved % 10 == 0:
                print(f"  {saved}/{n} (last: {cat_name})")

    print(f"Saved {saved} ESC-50 clips with high variety → {output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', choices=['common_voice', 'librispeech', 'esc50'],
                        default='common_voice')
    parser.add_argument('--n',      type=int, default=2000,
                        help='Number of clips to download (default 2000)')
    parser.add_argument('--output', default='sources/', help='Output directory')
    parser.add_argument('--lang',   default='en',       help='Language for Common Voice')
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    if args.dataset == 'common_voice':
        download_common_voice(out, args.n, args.lang)
    elif args.dataset == 'librispeech':
        download_librispeech(out, args.n)
    elif args.dataset == 'esc50':
        download_esc50(out, args.n)


if __name__ == '__main__':
    main()
