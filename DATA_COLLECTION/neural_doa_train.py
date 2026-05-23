import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf


DEFAULT_VOICE_DIR = Path("sessions/session_20260521_005414/recordings/voice")
OUT_DIRNAME = "neural_doa"
SAMPLE_RATE = 16000
MAX_TAU = 25
N_CLASSES = 72
ANGLE_STEP = 5
PAIRS_6 = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


def parse_truth(path):
    match = re.match(r"doa_(\d{3})(?:_\d+)?\.wav$", path.name)
    if not match:
        return None
    return int(match.group(1))


def parse_recording_id(path):
    match = re.match(r"doa_\d{3}_(\d+)\.wav$", path.name)
    if not match:
        return None
    return int(match.group(1))


def find_labeled_wavs(root_dir):
    return [
        path for path in sorted(Path(root_dir).rglob("doa_*.wav"))
        if parse_truth(path) is not None
    ]


def circular_error_deg(pred, truth):
    return abs(((pred - truth + 180.0) % 360.0) - 180.0)


def circular_mean_deg(angles):
    radians = np.deg2rad(angles)
    mean_angle = np.arctan2(np.mean(np.sin(radians)), np.mean(np.cos(radians)))
    return float(np.rad2deg(mean_angle) % 360.0)


def gcc_phat_lags(x, y, max_tau=MAX_TAU):
    n = int(2 ** np.ceil(np.log2(len(x) + len(y))))
    x_fft = np.fft.rfft(x, n=n)
    y_fft = np.fft.rfft(y, n=n)
    cross = x_fft * np.conj(y_fft)
    cross /= np.maximum(np.abs(cross), 1e-12)
    cc = np.fft.irfft(cross, n=n)
    return np.concatenate((cc[-max_tau:], cc[:max_tau + 1]))


def extract_gcc_features(audio, frame_samples=4096, hop_samples=2048):
    """GCC-PHAT features inspired by the existing SLCNet feature code, reduced to one vector."""
    audio = np.asarray(audio[:, :4], dtype=np.float32)
    audio -= np.mean(audio, axis=0, keepdims=True)
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio /= peak

    if audio.shape[0] < frame_samples:
        audio = np.pad(audio, ((0, frame_samples - audio.shape[0]), (0, 0)))

    frame_feats = []
    for start in range(0, audio.shape[0] - frame_samples + 1, hop_samples):
        frame = audio[start:start + frame_samples]
        pair_feats = [gcc_phat_lags(frame[:, i], frame[:, j]) for i, j in PAIRS_6]
        frame_feats.append(np.concatenate(pair_feats))

    feat = np.mean(frame_feats, axis=0).astype(np.float32)
    feat = (feat - feat.mean()) / (feat.std() + 1e-8)
    return feat


def build_feature_cache(voice_dirs, out_dir, force=False):
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_path = out_dir / "features.npz"
    if feature_path.exists() and not force:
        return feature_path

    if isinstance(voice_dirs, (str, Path)):
        voice_dirs = [voice_dirs]

    wav_paths = []
    for voice_dir in voice_dirs:
        wav_paths.extend(find_labeled_wavs(voice_dir))
    wav_paths = sorted(wav_paths)

    features = []
    labels = []
    files = []
    groups = []

    for idx, wav_path in enumerate(wav_paths, start=1):
        truth = parse_truth(wav_path)
        if truth is None:
            continue
        audio, fs = sf.read(wav_path)
        if fs != SAMPLE_RATE:
            raise ValueError(f"{wav_path} has fs={fs}, expected {SAMPLE_RATE}")
        if audio.ndim != 2 or audio.shape[1] < 4:
            continue

        features.append(extract_gcc_features(audio))
        label = truth // ANGLE_STEP
        labels.append(label)
        recording_id = parse_recording_id(wav_path)
        if recording_id is not None:
            group = f"{wav_path.parent}:{recording_id - label}"
        else:
            group = str(wav_path.parent)
        groups.append(group)
        files.append(str(wav_path))

        if idx % 100 == 0:
            print(f"Extracted features for {idx}/{len(wav_paths)} files")

    np.savez_compressed(
        feature_path,
        features=np.asarray(features, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        groups=np.asarray(groups),
        files=np.asarray(files),
    )
    return feature_path


def stratified_split(labels, test_ratio=0.2, seed=7):
    rng = np.random.default_rng(seed)
    train_idx = []
    test_idx = []
    for cls in sorted(set(labels.tolist())):
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        n_test = max(1, int(round(len(idx) * test_ratio)))
        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])
    return np.asarray(train_idx), np.asarray(test_idx)


def grouped_split(groups, test_ratio=0.2, seed=7):
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    rng.shuffle(unique_groups)
    n_test = max(1, int(round(len(unique_groups) * test_ratio)))
    test_groups = set(unique_groups[:n_test].tolist())
    test_mask = np.asarray([g in test_groups for g in groups])
    return np.where(~test_mask)[0], np.where(test_mask)[0]


def train_and_evaluate(feature_path, out_dir, epochs=80, batch_size=64, lr=1e-3, seed=7):
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "PyTorch is not installed. Install it, then rerun:\n"
            "  pip install torch\n"
            "  python neural_doa_train.py"
        ) from exc

    torch.manual_seed(seed)
    data = np.load(feature_path, allow_pickle=True)
    x = data["features"]
    y = data["labels"]
    groups = data["groups"] if "groups" in data else np.arange(len(y))
    files = data["files"]

    train_idx, test_idx = grouped_split(groups, seed=seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Split by sweep group: train={len(train_idx)} files, test={len(test_idx)} files, "
        f"train_groups={len(np.unique(groups[train_idx]))}, test_groups={len(np.unique(groups[test_idx]))}"
    )

    train_ds = TensorDataset(torch.from_numpy(x[train_idx]), torch.from_numpy(y[train_idx]))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = nn.Sequential(
        nn.Linear(x.shape[1], 128),
        nn.ReLU(),
        nn.Dropout(0.15),
        nn.Linear(128, 72),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for feat, target in train_loader:
            feat = feat.to(device)
            target = target.to(device)
            pred = model(feat)
            loss = loss_fn(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * feat.shape[0]

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            metrics = evaluate_model(model, x[test_idx], y[test_idx], device)
            print(
                f"Epoch {epoch:3d}/{epochs} "
                f"loss={total_loss / len(train_ds):.4f} "
                f"test_MAE={metrics['mae']:.2f}deg "
                f"test_RMSE={metrics['rmse']:.2f}deg "
                f"acc_5deg={metrics['acc_5deg']:.1f}%"
            )

    model_path = out_dir / "lightweight_gcc_mlp.pt"
    torch.save(model.state_dict(), model_path)

    pred_rows, angle_rows, metrics = make_eval_tables(
        model, x[test_idx], y[test_idx], files[test_idx], device
    )
    write_outputs(out_dir, pred_rows, angle_rows)
    print(f"Saved: {model_path}")
    return metrics


def load_trained_model(model_path, input_dim):
    try:
        import torch
        import torch.nn as nn
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "PyTorch is not installed. Install it, then rerun:\n"
            "  pip install torch"
        ) from exc

    model = nn.Sequential(
        nn.Linear(input_dim, 128),
        nn.ReLU(),
        nn.Dropout(0.15),
        nn.Linear(128, 72),
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model, torch.device("cpu")


def test_existing_model(feature_path, model_path, out_dir):
    data = np.load(feature_path, allow_pickle=True)
    x = data["features"]
    y = data["labels"]
    files = data["files"]
    model, device = load_trained_model(model_path, x.shape[1])
    pred_rows, angle_rows, metrics = make_eval_tables(model, x, y, files, device)
    write_outputs(out_dir, pred_rows, angle_rows)
    return metrics


def evaluate_model(model, x, y, device):
    import torch

    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(x).to(device))
        pred_cls = logits.argmax(dim=1).cpu().numpy()

    truth_deg = y * ANGLE_STEP
    pred_deg = pred_cls * ANGLE_STEP
    errors = np.asarray([circular_error_deg(p, t) for p, t in zip(pred_deg, truth_deg)])
    return {
        "mse": float(np.mean(errors ** 2)),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "mae": float(np.mean(errors)),
        "acc_5deg": float(np.mean(errors <= 5.0) * 100.0),
    }


def make_eval_tables(model, x, y, files, device):
    import torch

    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(x).to(device))
        pred_cls = logits.argmax(dim=1).cpu().numpy()

    pred_rows = []
    grouped = defaultdict(list)
    for file_name, truth_cls, pred_cls_i in zip(files, y, pred_cls):
        truth = int(truth_cls) * ANGLE_STEP
        pred = int(pred_cls_i) * ANGLE_STEP
        err = circular_error_deg(pred, truth)
        grouped[truth].append((pred, err))
        pred_rows.append({
            "file": str(file_name),
            "truth_deg": truth,
            "pred_deg": pred,
            "abs_circular_error_deg": err,
        })

    angle_rows = []
    for truth in sorted(grouped):
        preds = np.asarray([p for p, _ in grouped[truth]], dtype=np.float64)
        errs = np.asarray([e for _, e in grouped[truth]], dtype=np.float64)
        angle_rows.append({
            "truth_deg": truth,
            "count": len(errs),
            "mean_pred_deg": circular_mean_deg(preds),
            "mse_deg2": float(np.mean(errs ** 2)),
            "rmse_deg": float(np.sqrt(np.mean(errs ** 2))),
            "mae_deg": float(np.mean(errs)),
        })

    metrics = evaluate_model(model, x, y, device)
    return pred_rows, angle_rows, metrics


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(out_dir, pred_rows, angle_rows):
    pred_csv = out_dir / "neural_doa_predictions.csv"
    angle_csv = out_dir / "neural_doa_by_angle.csv"
    doa_plot = out_dir / "neural_doa_mean_by_angle.png"
    mse_plot = out_dir / "neural_doa_mse_by_angle.png"

    write_csv(pred_csv, pred_rows, ["file", "truth_deg", "pred_deg", "abs_circular_error_deg"])
    write_csv(angle_csv, angle_rows, ["truth_deg", "count", "mean_pred_deg", "mse_deg2", "rmse_deg", "mae_deg"])

    truth = [r["truth_deg"] for r in angle_rows]
    mean_pred = [r["mean_pred_deg"] for r in angle_rows]
    mse = [r["mse_deg2"] for r in angle_rows]

    plt.figure(figsize=(14, 6))
    ideal = np.arange(0, 360, 5)
    plt.plot(ideal, ideal, color="black", linestyle="--", linewidth=1.2, label="ideal")
    plt.plot(truth, mean_pred, marker="o", linewidth=1.5, markersize=3, label="lightweight GCC-MLP")
    plt.xlabel("Ground-truth DoA (deg)")
    plt.ylabel("Mean estimated DoA (deg)")
    plt.title("Neural DoA mean estimate per recorded angle")
    plt.xticks(np.arange(0, 360, 20))
    plt.yticks(np.arange(0, 360, 20))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(doa_plot, dpi=150)
    plt.close()

    plt.figure(figsize=(14, 5))
    plt.plot(truth, mse, marker="o", linewidth=1.5, markersize=3)
    plt.xlabel("Ground-truth DoA (deg)")
    plt.ylabel("MSE (deg^2)")
    plt.title("Neural DoA MSE per recorded angle")
    plt.xticks(np.arange(0, 360, 20))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(mse_plot, dpi=150)
    plt.close()

    print(f"Saved: {pred_csv}")
    print(f"Saved: {angle_csv}")
    print(f"Saved: {doa_plot}")
    print(f"Saved: {mse_plot}")


def main():
    parser = argparse.ArgumentParser(description="Train a lightweight neural DoA model.")
    parser.add_argument("--voice-dir", type=Path, action="append", default=None,
                        help="Recording root to scan. Can be passed multiple times.")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--force-features", action="store_true")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model-path", type=Path, default=None,
                        help="Evaluate an existing model instead of training.")
    args = parser.parse_args()

    voice_dirs = args.voice_dir or [DEFAULT_VOICE_DIR]
    out_dir = args.out_dir or (voice_dirs[0] / OUT_DIRNAME)
    feature_path = build_feature_cache(voice_dirs, out_dir, force=args.force_features)
    if args.model_path:
        metrics = test_existing_model(feature_path, args.model_path, out_dir)
    else:
        metrics = train_and_evaluate(
            feature_path,
            out_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
        )

    print(
        f"Final held-out metrics: MSE={metrics['mse']:.2f} deg^2, "
        f"RMSE={metrics['rmse']:.2f} deg, MAE={metrics['mae']:.2f} deg, "
        f"ACC<=5deg={metrics['acc_5deg']:.1f}%"
    )


if __name__ == "__main__":
    main()
