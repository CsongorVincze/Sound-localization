import torch
import torch.nn.functional as F
import torchaudio
import numpy as np

SAMPLE_RATE = 16000
FRAME_SAMPLES = int(SAMPLE_RATE * 0.170)  # 2720 - 170ms per frame
T_MAX = 27                                 # paper-faithful time dimension
GCC_DIM = 306                              # 6 pairs * 51 lags
MFCC_DIM = 312                             # 39 coeffs * 8 frames
FEAT_DIM = GCC_DIM + MFCC_DIM             # 618

# Module-level transforms (CPU); instantiated once per process
_mfcc = torchaudio.transforms.MFCC(
    sample_rate=SAMPLE_RATE,
    n_mfcc=13,
    melkwargs={'n_fft': 320, 'hop_length': 320, 'n_mels': 40}
)
_delta = torchaudio.transforms.ComputeDeltas()


def compute_gcc_phat(waveforms, max_tau=25):
    """
    waveforms: (1, 4, T)
    Returns:   (306,) - 6 mic pairs x 51 lag bins
    """
    _, _, T = waveforms.shape
    X = torch.fft.rfft(waveforms, dim=-1)
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    out = []
    for i, j in pairs:
        R = X[:, i, :] * torch.conj(X[:, j, :])
        R_phat = R / (torch.abs(R) + 1e-8)
        cc = torch.fft.irfft(R_phat, n=T, dim=-1)
        # lags: [-max_tau, ..., 0, ..., max_tau] -> 51 values
        out.append(torch.cat((cc[:, -max_tau:], cc[:, :max_tau + 1]), dim=-1))
    return torch.cat(out, dim=-1).squeeze(0)  # (306,)


def extract_frame_features(chunk):
    """
    chunk: (1, 4, FRAME_SAMPLES)
    Returns: (618,) - GCC-PHAT (306) + MFCC+delta+delta2 (312)
    """
    gcc = compute_gcc_phat(chunk)  # (306,)

    mono = chunk.mean(dim=1)                          # (1, FRAME_SAMPLES)
    mfcc = _mfcc(mono)                                # (1, 13, ~8)
    d1 = _delta(mfcc)
    d2 = _delta(d1)
    mfcc_all = torch.cat([mfcc, d1, d2], dim=1).squeeze(0)  # (39, ~8)
    # adaptive pool guarantees exactly 8 frames regardless of torchaudio padding
    mfcc_all = F.adaptive_avg_pool1d(mfcc_all.unsqueeze(0), 8).squeeze(0)  # (39, 8)

    return torch.cat([gcc, mfcc_all.reshape(-1)], dim=0)  # (618,)


def normalize_features(feat):
    """
    feat: (1, T, 618)
    Min-max normalize GCC-PHAT and MFCC separately along feature axis (dim=2),
    matching paper's minmax_norm2d(faxis=2).
    """
    gcc = feat[:, :, :GCC_DIM]
    mfcc = feat[:, :, GCC_DIM:]

    def _minmax(x):
        lo = x.min(dim=2, keepdim=True).values
        hi = x.max(dim=2, keepdim=True).values
        return (x - lo) / (hi - lo + 1e-8)

    return torch.cat([_minmax(gcc), _minmax(mfcc)], dim=2)


def make_doa_target(angle_deg, sigma=6.0):
    """
    angle_deg: integer DoA in 1..360
    Returns: (360,) float32 Gaussian with circular wrap, unnormalized (peak = 1)
    sigma=6 from He et al. 2018 (cited in SLoClas paper for likelihood-based coding)
    """
    angles = np.arange(1, 361, dtype=np.float32)
    diff = np.abs(angles - float(angle_deg))
    diff = np.minimum(diff, 360.0 - diff)  # circular angular distance
    return np.exp(-(diff ** 2) / (sigma ** 2)).astype(np.float32)


def extract_file_features(audio_4ch):
    """
    audio_4ch: (4, T) float32 tensor at SAMPLE_RATE, amplitude-normalized
    Returns: (T_MAX, 618) float32 numpy array, zero-padded where audio is shorter than T_MAX
    """
    n_frames = min(audio_4ch.shape[1] // FRAME_SAMPLES, T_MAX)

    frame_feats = []
    for t in range(n_frames):
        chunk = audio_4ch[:, t * FRAME_SAMPLES:(t + 1) * FRAME_SAMPLES].unsqueeze(0)
        frame_feats.append(extract_frame_features(chunk))

    # Zero-pad remaining frames so MaxPool sees clean zeros (not MFCC-of-silence)
    for _ in range(T_MAX - n_frames):
        frame_feats.append(torch.zeros(FEAT_DIM))

    feat = torch.stack(frame_feats).unsqueeze(0)  # (1, T_MAX, 618)
    feat = normalize_features(feat).squeeze(0)    # (T_MAX, 618)
    return feat.numpy()
