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


class DoANet(nn.Module):
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
    n = int(2 ** np.ceil(np.log2(len(x) + len(y))))
    x_fft = np.fft.rfft(x, n=n)
    y_fft = np.fft.rfft(y, n=n)
    cross = x_fft * np.conj(y_fft)
    cross /= np.maximum(np.abs(cross), 1e-12)
    cc = np.fft.irfft(cross, n=n)
    return np.concatenate((cc[-MAX_TAU:], cc[:MAX_TAU + 1]))


def extract_features(audio, frame_samples=4096, hop_samples=2048):
    audio = np.asarray(audio[:, :4], dtype=np.float32)
    audio -= audio.mean(axis=0, keepdims=True)
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio /= peak

    if len(audio) < frame_samples:
        audio = np.pad(audio, ((0, frame_samples - len(audio)), (0, 0)))

    frames = []
    for start in range(0, len(audio) - frame_samples + 1, hop_samples):
        frame = audio[start:start + frame_samples]
        pair_features = [gcc_phat_lags(frame[:, i], frame[:, j]) for i, j in PAIRS]
        frames.append(np.concatenate(pair_features))

    feature = np.mean(frames, axis=0).astype(np.float32)
    return (feature - feature.mean()) / (feature.std() + 1e-8)


def load_or_extract_dataset(roots, out_dir, force=False):
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "features.npz"
    if cache_path.exists() and not force:
        data = np.load(cache_path, allow_pickle=True)
        return data["x"], data["y"], data["groups"], data["files"]

    x, y, groups, files = [], [], [], []
    recordings = find_recordings(roots)
    for index, path in enumerate(recordings, start=1):
        angle = angle_from_name(path)
        audio, fs = sf.read(path)
        if fs != SAMPLE_RATE or audio.ndim != 2 or audio.shape[1] < 4:
            continue

        label = angle // ANGLE_STEP
        x.append(extract_features(audio))
        y.append(label)
        groups.append(recording_group(path, label))
        files.append(str(path))

        if index % 100 == 0:
            print(f"features: {index}/{len(recordings)}")

    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    groups = np.asarray(groups)
    files = np.asarray(files)
    np.savez_compressed(cache_path, x=x, y=y, groups=groups, files=files)
    return x, y, groups, files


def split_by_group(groups, test_ratio=0.2, seed=7):
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    rng.shuffle(unique_groups)
    test_count = max(1, round(len(unique_groups) * test_ratio))
    test_groups = set(unique_groups[:test_count])
    is_test = np.asarray([group in test_groups for group in groups])
    return np.where(~is_test)[0], np.where(is_test)[0]


def circular_error(pred_deg, true_deg):
    return abs(((pred_deg - true_deg + 180) % 360) - 180)


def evaluate(model, x, y, files, device):
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(x).to(device))
        pred_labels = logits.argmax(dim=1).cpu().numpy()

    if files is None or len(files) == 0:
        files = [""] * len(y)

    rows = []
    for file_name, true_label, pred_label in zip(files, y, pred_labels):
        true_deg = int(true_label) * ANGLE_STEP
        pred_deg = int(pred_label) * ANGLE_STEP
        rows.append({
            "file": file_name,
            "truth_deg": true_deg,
            "pred_deg": pred_deg,
            "error_deg": circular_error(pred_deg, true_deg),
        })

    predictions = pd.DataFrame(rows)
    errors = predictions["error_deg"].to_numpy(dtype=float)
    metrics = {
        "mse": float(np.mean(errors ** 2)),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "mae": float(np.mean(errors)),
        "acc_5deg": float(np.mean(errors <= 5) * 100),
    }
    by_angle = predictions.groupby("truth_deg")["error_deg"].agg(
        count="count",
        mse_deg2=lambda s: float(np.mean(s ** 2)),
        rmse_deg=lambda s: float(np.sqrt(np.mean(s ** 2))),
        mae_deg="mean",
    ).reset_index()
    return metrics, predictions, by_angle


def train_model(x, y, train_idx, test_idx, epochs, batch_size, lr, seed):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DoANet(x.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    train_data = TensorDataset(torch.from_numpy(x[train_idx]), torch.from_numpy(y[train_idx]))
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            loss = loss_fn(model(features), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            metrics, _, _ = evaluate(model, x[test_idx], y[test_idx], [], device)
            print(
                f"epoch {epoch:3d}/{epochs} "
                f"loss={total_loss / len(train_data):.4f} "
                f"mae={metrics['mae']:.2f}deg "
                f"rmse={metrics['rmse']:.2f}deg "
                f"acc5={metrics['acc_5deg']:.1f}%"
            )

    return model, device


def save_evaluation(out_dir, model, x, y, files, device):
    metrics, predictions, by_angle = evaluate(model, x, y, files, device)
    predictions.to_csv(out_dir / "predictions.csv", index=False)
    by_angle.to_csv(out_dir / "by_angle.csv", index=False)
    pd.DataFrame([metrics]).to_csv(out_dir / "metrics.csv", index=False)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train/evaluate a simple GCC-PHAT DoA network.")
    parser.add_argument("--data-dir", type=Path, action="append", default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("sessions/simple_neural_doa"))
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--force-features", action="store_true")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    roots = args.data_dir or DEFAULT_DIRS
    x, y, groups, files = load_or_extract_dataset(roots, args.out_dir, args.force_features)
    train_idx, test_idx = split_by_group(groups, seed=args.seed)
    print(f"dataset: {len(x)} files, train={len(train_idx)}, test={len(test_idx)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model_path:
        model = DoANet(x.shape[1]).to(device)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    else:
        model, device = train_model(
            x, y, train_idx, test_idx,
            args.epochs, args.batch_size, args.lr, args.seed,
        )
        torch.save(model.state_dict(), args.out_dir / "model.pt")

    metrics = save_evaluation(args.out_dir, model, x[test_idx], y[test_idx], files[test_idx], device)
    print(
        f"test: mse={metrics['mse']:.2f}, rmse={metrics['rmse']:.2f}deg, "
        f"mae={metrics['mae']:.2f}deg, acc5={metrics['acc_5deg']:.1f}%"
    )
    print(f"saved outputs in: {args.out_dir}")


if __name__ == "__main__":
    main()
