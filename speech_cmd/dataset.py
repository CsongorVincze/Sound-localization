"""
Google Speech Commands v2 dataset loader.

Expected layout:
    <root>/
        _background_noise_/   (6 long .wav noise recordings)
        yes/ no/ go/ stop/ ...  (35 word directories)
        validation_list.txt
        testing_list.txt

24 robot command classes + silence + unknown = 26 total.
Unknown words are sub-sampled to match the average per-target-class count.
Silence is generated from background noise at the same count.
"""

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from pathlib import Path
from torch.utils.data import Dataset

SAMPLE_RATE  = 16000
CLIP_SAMPLES = 16000  # 1 second at 16 kHz

TARGET_CLASSES = [
    'backward', 'down', 'eight', 'five', 'follow', 'forward',
    'four', 'go', 'learn', 'left', 'nine', 'no', 'off', 'on',
    'one', 'right', 'seven', 'six', 'stop', 'three', 'two',
    'up', 'yes', 'zero',
]
CLASS_NAMES = TARGET_CLASSES + ['silence', 'unknown']
LABEL_MAP   = {name: i for i, name in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)  # 26

# Module-level transforms: stateless, safe to share across workers
_mel = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=480,
    win_length=480,
    hop_length=160,   # 10 ms hop -> (16000 + 480 - 480) / 160 + 1 = 101 frames
    n_mels=40,
    f_min=20.0,
)
_to_db = torchaudio.transforms.AmplitudeToDB(top_db=80)


def wav_to_spec(wav: torch.Tensor) -> torch.Tensor:
    """wav: (CLIP_SAMPLES,) float32 -> spec: (1, 40, 101), per-sample z-normalised."""
    spec = _to_db(_mel(wav.unsqueeze(0)))
    return (spec - spec.mean()) / (spec.std() + 1e-8)


class SpeechCommandsDataset(Dataset):
    def __init__(self, root: str, split: str = 'train', augment: bool = False):
        assert split in ('train', 'val', 'test'), f"Unknown split: {split}"
        root    = Path(root)
        augment = augment and split == 'train'
        self.augment = augment

        val_files  = set((root / 'validation_list.txt').read_text().splitlines())
        test_files = set((root / 'testing_list.txt').read_text().splitlines())

        target_samples  = []
        unknown_samples = []

        for word_dir in sorted(d for d in root.iterdir()
                               if d.is_dir() and not d.name.startswith('_')):
            word  = word_dir.name
            label = LABEL_MAP.get(word, LABEL_MAP['unknown'])
            for wav_path in sorted(word_dir.glob('*.wav')):
                rel     = f"{word}/{wav_path.name}"
                in_val  = rel in val_files
                in_test = rel in test_files
                keep = (
                    (split == 'val'   and in_val)                       or
                    (split == 'test'  and in_test)                      or
                    (split == 'train' and not in_val and not in_test)
                )
                if not keep:
                    continue
                entry = (wav_path, label, False)  # (path, label, is_noise)
                if label == LABEL_MAP['unknown']:
                    unknown_samples.append(entry)
                else:
                    target_samples.append(entry)

        # Sub-sample unknown to match average per-target-class count for class balance
        n_per_class = len(target_samples) // max(1, len(TARGET_CLASSES))
        rng = np.random.default_rng(seed=42)
        if len(unknown_samples) > n_per_class:
            idxs            = rng.choice(len(unknown_samples), n_per_class, replace=False)
            unknown_samples = [unknown_samples[i] for i in sorted(idxs)]

        # Silence: random 1s crops from background noise files
        noise_dir  = root / '_background_noise_'
        noise_wavs = sorted(noise_dir.glob('*.wav'))
        if not noise_wavs:
            raise FileNotFoundError(f"No .wav files in {noise_dir}")

        silence_samples = []
        for _ in range(n_per_class):
            silence_samples.append((
                noise_wavs[rng.integers(len(noise_wavs))],
                LABEL_MAP['silence'],
                True,   # is_noise -> random crop in __getitem__
            ))

        self.samples = target_samples + unknown_samples + silence_samples

        if augment:
            self._freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=10)
            self._time_mask = torchaudio.transforms.TimeMasking(time_mask_param=25)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label, is_noise = self.samples[idx]

        import soundfile as sf
        data, sr = sf.read(str(path), dtype='float32')
        wav = torch.from_numpy(data)
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        else:
            wav = wav.transpose(0, 1)
        if sr != SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
        wav = wav.mean(dim=0)  # multi-channel -> mono

        if is_noise:
            max_start = max(0, wav.shape[0] - CLIP_SAMPLES)
            start = int(torch.randint(0, max_start + 1, (1,)).item()) if max_start > 0 else 0
            wav = wav[start : start + CLIP_SAMPLES]

        if wav.shape[0] < CLIP_SAMPLES:
            wav = F.pad(wav, (0, CLIP_SAMPLES - wav.shape[0]))
        else:
            wav = wav[:CLIP_SAMPLES]

        if self.augment:
            wav = (wav * torch.empty(1).uniform_(0.5, 1.5)).clamp(-1.0, 1.0)

        spec = wav_to_spec(wav)

        if self.augment:
            spec = self._freq_mask(spec)
            spec = self._time_mask(spec)

        return spec, label
