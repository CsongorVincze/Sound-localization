"""
Synthetic 4-channel data generation using pyroomacoustics + ESC-50.

Usage:
    python simulate.py --esc50_dir /path/to/ESC-50 --out_dir ./sim_preprocessed
    python simulate.py --esc50_dir /path/to/ESC-50 --out_dir ./sim_preprocessed --num_workers 8

Output format is identical to preprocess.py — drop-in for train.py:
    python train.py --feat_dir ./sim_preprocessed --num_classes 50

Runtime estimate: ~144K simulations. Single-threaded: ~8-12h. With 8 workers: ~1-2h.
"""

import argparse
import multiprocessing as mp
import numpy as np
import pandas as pd
import pyroomacoustics as pra
import soundfile as sf
import torch
import torchaudio
from pathlib import Path
from tqdm import tqdm

from features import SAMPLE_RATE, T_MAX, FEAT_DIM, extract_file_features, make_doa_target

# ReSpeaker v2.0 geometry: 4 MEMS mics, 35mm radius, 90-deg spacing
MIC_RADIUS     = 0.035
MIC_ANGLES_RAD = np.deg2rad([0, 90, 180, 270])
ROOM_DIM       = [5.0, 4.0, 3.0]
MIC_CENTER     = [ROOM_DIM[0] / 2, ROOM_DIM[1] / 2, 1.0]
SOURCE_DIST    = 1.5  # metres from mic-array centre to source
MAX_ORDER      = 3    # image-source reflections; 3 balances accuracy vs speed

# Pre-compute mic positions once (workers inherit via module import with spawn)
_MIC_POS = np.vstack([
    MIC_CENTER[0] + MIC_RADIUS * np.cos(MIC_ANGLES_RAD),
    MIC_CENTER[1] + MIC_RADIUS * np.sin(MIC_ANGLES_RAD),
    np.full(4, MIC_CENTER[2])
])


def _simulate(audio_mono, doa_deg, rt60):
    """Simulate one 4-channel recording via image-source method."""
    e_abs, _ = pra.inverse_sabine(rt60, ROOM_DIM)
    room = pra.ShoeBox(
        ROOM_DIM, fs=SAMPLE_RATE,
        materials=pra.Material(e_abs),
        max_order=MAX_ORDER
    )
    room.add_microphone(_MIC_POS.copy())
    phi = np.deg2rad(doa_deg)
    src = [
        MIC_CENTER[0] + SOURCE_DIST * np.cos(phi),
        MIC_CENTER[1] + SOURCE_DIST * np.sin(phi),
        MIC_CENTER[2],
    ]
    room.add_source(src, signal=audio_mono.astype(np.float64))
    room.simulate()
    return room.mic_array.signals.astype(np.float32)  # (4, T)


def _load_clip(wav_path):
    """Load mono wav via soundfile, resample to SAMPLE_RATE, peak-normalise."""
    data, sr = sf.read(str(wav_path), always_2d=True)
    audio = data[:, 0].astype(np.float32)
    if sr != SAMPLE_RATE:
        t = torch.from_numpy(audio).unsqueeze(0)
        audio = torchaudio.functional.resample(t, sr, SAMPLE_RATE).squeeze(0).numpy()
    peak = np.abs(audio).max()
    if peak > 0:
        audio /= peak
    return audio


def _process_clip(task):
    """Worker function: simulate all angles for one ESC-50 clip.

    Returns:
        results: list of (feat, doa_target, class_idx, doa_deg) or None per angle
        errors:  list of error strings or None per angle
    """
    wav_path, class_idx, angles, rt60_min, rt60_max, seed = task
    rng = np.random.default_rng(seed)
    results, errors = [], []

    try:
        audio_mono = _load_clip(wav_path)
    except Exception as e:
        n = len(angles)
        return [None] * n, [f"load:{wav_path}: {e}"] * n

    for doa_deg in angles:
        rt60 = rng.uniform(rt60_min, rt60_max)
        try:
            mc = _simulate(audio_mono, doa_deg, rt60)
            audio_4ch = torch.from_numpy(mc)
            peak = audio_4ch.abs().max()
            if peak > 0:
                audio_4ch = audio_4ch / peak
            feat  = extract_file_features(audio_4ch)
            doa_t = make_doa_target(doa_deg, sigma=6.0)
            results.append((feat, doa_t, class_idx, doa_deg))
            errors.append(None)
        except Exception as e:
            results.append(None)
            errors.append(f"{wav_path}@{doa_deg}deg: {e}")

    return results, errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--esc50_dir', required=True,
                        help='ESC-50 root directory (contains audio/ and meta/esc50.csv)')
    parser.add_argument('--out_dir',     default='./sim_preprocessed')
    parser.add_argument('--rt60_min',    type=float, default=0.2,
                        help='Minimum RT60 reverberation time in seconds')
    parser.add_argument('--rt60_max',    type=float, default=0.6,
                        help='Maximum RT60 reverberation time in seconds')
    parser.add_argument('--n_angles',    type=int,   default=72,
                        help='DoA angles uniformly spaced 5..360 (72 = 5-deg step)')
    parser.add_argument('--seed',        type=int,   default=42)
    parser.add_argument('--num_workers', type=int,   default=1,
                        help='Parallel worker processes (recommended: 4-16 on a server)')
    args = parser.parse_args()

    step   = 360 // args.n_angles
    angles = list(range(step, 361, step))  # e.g. [5, 10, ..., 360] for n_angles=72

    esc50_dir = Path(args.esc50_dir)
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta        = pd.read_csv(esc50_dir / 'meta' / 'esc50.csv')
    clips       = [(str(esc50_dir / 'audio' / row.filename), int(row.target))
                   for _, row in meta.iterrows()]
    num_classes = meta['target'].nunique()
    N           = len(clips) * len(angles)

    print(f"ESC-50: {len(clips)} clips, {num_classes} classes, "
          f"{len(angles)} angles ({step}-deg step) -> {N} samples")
    print(f"Features array: {N * T_MAX * FEAT_DIM * 4 / 1e9:.1f} GB — ensure sufficient RAM")
    print(f"Workers: {args.num_workers}")

    tasks = [
        (wav, cls, angles, args.rt60_min, args.rt60_max, args.seed * 10000 + i)
        for i, (wav, cls) in enumerate(clips)
    ]

    features       = np.zeros((N, T_MAX, FEAT_DIM), dtype=np.float32)
    doa_targets    = np.zeros((N, 360),             dtype=np.float32)
    class_labels   = np.zeros(N,                    dtype=np.int64)
    doa_angles_out = np.zeros(N,                    dtype=np.int16)

    all_errors = []
    write_idx  = 0
    pool       = None

    if args.num_workers > 1:
        ctx  = mp.get_context('spawn')
        pool = ctx.Pool(args.num_workers)
        imap = pool.imap(_process_clip, tasks, chunksize=2)
    else:
        imap = map(_process_clip, tasks)

    try:
        for clip_results, clip_errors in tqdm(imap, total=len(clips),
                                              desc='Clips', unit='clip'):
            for r, e in zip(clip_results, clip_errors):
                if r is not None:
                    feat, doa_t, cls_idx, doa_deg = r
                    features[write_idx]       = feat
                    doa_targets[write_idx]    = doa_t
                    class_labels[write_idx]   = cls_idx
                    doa_angles_out[write_idx] = doa_deg
                    write_idx += 1
                if e is not None:
                    all_errors.append(e)
    except KeyboardInterrupt:
        print("\nInterrupted — saving partial results...")
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()

    if write_idx == 0:
        raise RuntimeError("No samples succeeded — nothing to save.")

    if all_errors:
        error_rate = len(all_errors) / N
        print(f"\n{len(all_errors)}/{N} samples failed ({error_rate*100:.1f}%):")
        for e in all_errors[:10]:
            print(f"  {e}")
        if error_rate > 0.05:
            print(f"Warning: high failure rate ({error_rate*100:.1f}%). "
                  f"Check pyroomacoustics install and ESC-50 paths.")

    np.save(out_dir / 'features.npy',     features[:write_idx])
    np.save(out_dir / 'doa_targets.npy',  doa_targets[:write_idx])
    np.save(out_dir / 'class_labels.npy', class_labels[:write_idx])
    np.save(out_dir / 'doa_angles.npy',   doa_angles_out[:write_idx])

    saved = features[:write_idx]
    print(f"\nSaved {write_idx} samples to {out_dir}/")
    print(f"  features.npy:     {saved.shape}  ({saved.nbytes/1e9:.1f} GB)")
    print(f"  doa_targets.npy:  {doa_targets[:write_idx].shape}")
    print(f"  class_labels.npy: {class_labels[:write_idx].shape}")
    print(f"\nRun training:")
    print(f"  python train.py --feat_dir {out_dir} --num_classes {num_classes}")


if __name__ == '__main__':
    main()
