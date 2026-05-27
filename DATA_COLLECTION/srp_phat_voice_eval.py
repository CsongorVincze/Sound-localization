import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf


DEFAULT_VOICE_DIR = Path(
    "sessions/session_20260521_005414/recordings/voice"
)
PICTURE_DIR = Path(__file__).resolve().parent / "evaluation_pictures"
RESULT_DIR = Path(__file__).resolve().parent / "evaluation_results"

SPEED_OF_SOUND = 343.0
MIC_RADIUS_M = 0.035
NFFT = 1024
HOP = 512
FREQ_MIN = 300.0
FREQ_MAX = 4500.0

PAIR_SCENARIOS = {
    "2_pairs_opposite": [(0, 2), (1, 3)],
    "4_pairs_perimeter": [(0, 1), (1, 2), (2, 3), (3, 0)],
    "6_pairs_all": [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
}


def respeaker_mic_positions(radius_m=MIC_RADIUS_M):
    """4-mic ReSpeaker v2.0 square-corner geometry, channels 0..3."""
    mic_angles = np.deg2rad([-45.0, 45.0, 135.0, 225.0])
    return np.column_stack((radius_m * np.cos(mic_angles), radius_m * np.sin(mic_angles)))


def circular_error_deg(pred, truth):
    return abs(((pred - truth + 180.0) % 360.0) - 180.0)


def frame_signal(audio, nfft=NFFT, hop=HOP):
    if audio.shape[0] < nfft:
        pad = nfft - audio.shape[0]
        audio = np.pad(audio, ((0, pad), (0, 0)))

    n_frames = 1 + (audio.shape[0] - nfft) // hop
    frames = np.empty((n_frames, nfft, audio.shape[1]), dtype=np.float64)
    window = np.hanning(nfft)[:, None]

    for k in range(n_frames):
        start = k * hop
        frames[k] = audio[start:start + nfft] * window

    return frames


def srp_phat_doa(audio, fs, pairs, grid_degrees=None, mic_positions=None):
    if grid_degrees is None:
        grid_degrees = np.arange(360.0)
    if mic_positions is None:
        mic_positions = respeaker_mic_positions()

    audio = np.asarray(audio[:, :4], dtype=np.float64)
    audio -= np.mean(audio, axis=0, keepdims=True)

    frames = frame_signal(audio)
    spectra = np.fft.rfft(frames, n=NFFT, axis=1)
    freqs = np.fft.rfftfreq(NFFT, 1.0 / fs)
    freq_mask = (freqs >= FREQ_MIN) & (freqs <= FREQ_MAX)
    freqs = freqs[freq_mask]
    spectra = spectra[:, freq_mask, :]

    theta = np.deg2rad(grid_degrees)
    directions = np.column_stack((np.cos(theta), np.sin(theta)))
    response = np.zeros(len(grid_degrees), dtype=np.float64)

    for i, j in pairs:
        cross = spectra[:, :, i] * np.conj(spectra[:, :, j])
        cross /= np.maximum(np.abs(cross), 1e-12)
        mean_cross = np.mean(cross, axis=0)

        # Far-field TDOA for this pair over every candidate angle.
        tau = (mic_positions[i] - mic_positions[j]) @ directions.T / SPEED_OF_SOUND
        steering = np.exp(1j * 2.0 * np.pi * freqs[:, None] * tau[None, :])
        response += np.real(np.sum(mean_cross[:, None] * steering, axis=0))

    return float(grid_degrees[int(np.argmax(response))] % 360.0)


def parse_truth(path):
    match = re.match(r"doa_(\d{3})(?:_\d+)?\.wav$", path.name)
    if not match:
        return None
    return float(match.group(1))


def find_labeled_wavs(root_dir):
    return [
        path for path in sorted(Path(root_dir).rglob("doa_*.wav"))
        if parse_truth(path) is not None
    ]


def evaluate(voice_dir):
    wav_paths = find_labeled_wavs(voice_dir)
    rows = []
    errors_by_scenario_angle = {
        name: defaultdict(list) for name in PAIR_SCENARIOS
    }
    preds_by_scenario_angle = {
        name: defaultdict(list) for name in PAIR_SCENARIOS
    }

    for index, wav_path in enumerate(wav_paths, start=1):
        truth = parse_truth(wav_path)
        if truth is None:
            continue

        audio, fs = sf.read(wav_path)
        if audio.ndim != 2 or audio.shape[1] < 4:
            print(f"Skipping {wav_path.name}: expected at least 4 channels")
            continue

        row = {
            "file": wav_path.name,
            "truth_deg": truth,
        }
        for scenario, pairs in PAIR_SCENARIOS.items():
            pred = srp_phat_doa(audio, fs, pairs)
            err = circular_error_deg(pred, truth)
            errors_by_scenario_angle[scenario][truth].append(err)
            preds_by_scenario_angle[scenario][truth].append(pred)
            row[f"{scenario}_pred_deg"] = pred
            row[f"{scenario}_abs_circular_error_deg"] = err
        rows.append(row)

        if index % 100 == 0:
            print(f"Processed {index}/{len(wav_paths)} files")

    if not rows:
        raise RuntimeError(f"No valid doa_###_*.wav files found in {voice_dir}")

    angle_rows = []
    for scenario in PAIR_SCENARIOS:
        for angle in sorted(errors_by_scenario_angle[scenario]):
            errors = np.asarray(errors_by_scenario_angle[scenario][angle], dtype=np.float64)
            preds = np.asarray(preds_by_scenario_angle[scenario][angle], dtype=np.float64)
            angle_rows.append({
                "scenario": scenario,
                "truth_deg": angle,
                "count": int(errors.size),
                "mean_pred_deg": circular_mean_deg(preds),
                "mse_deg2": float(np.mean(errors ** 2)),
                "rmse_deg": float(np.sqrt(np.mean(errors ** 2))),
                "mae_deg": float(np.mean(errors)),
            })

    return rows, angle_rows


def circular_mean_deg(angles):
    radians = np.deg2rad(angles)
    mean_angle = np.arctan2(np.mean(np.sin(radians)), np.mean(np.cos(radians)))
    return float(np.rad2deg(mean_angle) % 360.0)


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def output_name_prefix(root_dir):
    parts = Path(root_dir).parts
    if "sessions" in parts:
        idx = parts.index("sessions")
        suffix = "_".join(parts[idx + 1:])
        return suffix or "sessions"
    return Path(root_dir).name or "evaluation"


def plot_mse(path, angle_rows):
    plt.figure(figsize=(14, 5))
    for scenario in PAIR_SCENARIOS:
        scenario_rows = [r for r in angle_rows if r["scenario"] == scenario]
        angles = [r["truth_deg"] for r in scenario_rows]
        mses = [r["mse_deg2"] for r in scenario_rows]
        plt.plot(angles, mses, marker="o", linewidth=1.5, markersize=3, label=scenario)

    plt.xlabel("Ground-truth DoA (deg)")
    plt.ylabel("MSE (deg^2)")
    plt.title("SRP-PHAT DoA error per angle - voice recordings")
    plt.xticks(np.arange(0, 360, 20))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_mean_doa(path, angle_rows):
    plt.figure(figsize=(14, 6))
    truth_line = np.arange(0.0, 360.0, 5.0)
    plt.plot(truth_line, truth_line, color="black", linestyle="--", linewidth=1.2, label="ideal")

    for scenario in PAIR_SCENARIOS:
        scenario_rows = [r for r in angle_rows if r["scenario"] == scenario]
        angles = [r["truth_deg"] for r in scenario_rows]
        preds = [r["mean_pred_deg"] for r in scenario_rows]
        plt.plot(angles, preds, marker="o", linewidth=1.5, markersize=3, label=scenario)

    plt.xlabel("Ground-truth DoA (deg)")
    plt.ylabel("Mean estimated DoA (deg)")
    plt.title("SRP-PHAT mean estimated DoA per recorded angle")
    plt.xticks(np.arange(0, 360, 20))
    plt.yticks(np.arange(0, 360, 20))
    plt.xlim(-2, 357)
    plt.ylim(-2, 357)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_mean_error_bars(path, angle_rows):
    angles = sorted({r["truth_deg"] for r in angle_rows})
    bar_width = 1.25
    offsets = np.linspace(
        -bar_width * (len(PAIR_SCENARIOS) - 1) / 2,
        bar_width * (len(PAIR_SCENARIOS) - 1) / 2,
        len(PAIR_SCENARIOS),
    )

    plt.figure(figsize=(18, 6))
    for offset, scenario in zip(offsets, PAIR_SCENARIOS):
        scenario_rows = {
            r["truth_deg"]: r for r in angle_rows
            if r["scenario"] == scenario
        }
        mean_errors = [scenario_rows[angle]["mae_deg"] for angle in angles]
        plt.bar(
            np.asarray(angles) + offset,
            mean_errors,
            width=bar_width,
            label=scenario,
        )

    plt.xlabel("Ground-truth DoA (deg)")
    plt.ylabel("Mean absolute error (deg)")
    plt.title("SRP-PHAT mean DoA error per angle - voice recordings")
    plt.xticks(np.arange(0, 360, 10), rotation=90)
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_mean_error_bar_per_scenario(picture_dir, prefix, angle_rows):
    hungarian_names = {
        "2_pairs_opposite": "2 mikrofonpár - szemközti párok",
        "4_pairs_perimeter": "4 mikrofonpár - szomszédos párok",
        "6_pairs_all": "6 mikrofonpár - összes pár",
    }
    colors = {
        "2_pairs_opposite": "#4477AA",
        "4_pairs_perimeter": "#EE6677",
        "6_pairs_all": "#228833",
    }
    output_paths = []
    max_error = max(
        r["mae_deg"] for r in angle_rows
        if r["scenario"] in PAIR_SCENARIOS
    )
    y_max = max(5.0, np.ceil((max_error + 0.5) / 2.0) * 2.0)
    y_ticks = np.arange(0.0, y_max + 0.1, 1.0)

    for scenario in PAIR_SCENARIOS:
        scenario_rows = [
            r for r in angle_rows
            if r["scenario"] == scenario
        ]
        angles = [r["truth_deg"] for r in scenario_rows]
        mean_errors = [r["mae_deg"] for r in scenario_rows]

        path = picture_dir / f"{prefix}_{scenario}_atlagos_hiba_oszlopdiagram.png"
        plt.figure(figsize=(16, 5))
        plt.bar(angles, mean_errors, width=3.5, color=colors[scenario])
        plt.xlabel("Valós beesési szög (fok)")
        plt.ylabel("Átlagos abszolút hiba (fok)")
        plt.title(f"SRP-PHAT átlagos DoA hiba szögenként - {hungarian_names[scenario]}")
        plt.xticks(np.arange(0, 360, 10), rotation=90)
        plt.ylim(0, y_max)
        plt.yticks(y_ticks)
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        output_paths.append(path)

    return output_paths


def main():
    parser = argparse.ArgumentParser(description="Evaluate SRP-PHAT DoA on voice recordings.")
    parser.add_argument("--voice-dir", type=Path, default=DEFAULT_VOICE_DIR)
    args = parser.parse_args()

    rows, angle_rows = evaluate(args.voice_dir)

    PICTURE_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    prefix = output_name_prefix(args.voice_dir)
    pred_csv = RESULT_DIR / f"{prefix}_srp_phat_predictions.csv"
    mse_csv = RESULT_DIR / f"{prefix}_srp_phat_mse_by_angle.csv"
    mse_plot_path = PICTURE_DIR / f"{prefix}_srp_phat_mse_by_angle.png"
    doa_plot_path = PICTURE_DIR / f"{prefix}_srp_phat_mean_doa_by_angle.png"

    pred_fields = ["file", "truth_deg"]
    for scenario in PAIR_SCENARIOS:
        pred_fields.extend([
            f"{scenario}_pred_deg",
            f"{scenario}_abs_circular_error_deg",
        ])
    write_csv(pred_csv, rows, pred_fields)
    write_csv(mse_csv, angle_rows, [
        "scenario", "truth_deg", "count", "mean_pred_deg",
        "mse_deg2", "rmse_deg", "mae_deg",
    ])
    plot_mse(mse_plot_path, angle_rows)
    plot_mean_doa(doa_plot_path, angle_rows)
    mean_error_bar_paths = plot_mean_error_bar_per_scenario(
        PICTURE_DIR,
        prefix,
        angle_rows,
    )

    print(f"Files processed: {len(rows)}")
    print(f"Angles processed: {len({r['truth_deg'] for r in angle_rows})}")
    for scenario in PAIR_SCENARIOS:
        all_errors = np.asarray(
            [r[f"{scenario}_abs_circular_error_deg"] for r in rows],
            dtype=np.float64,
        )
        print(
            f"{scenario}: MSE={np.mean(all_errors ** 2):.2f} deg^2, "
            f"RMSE={np.sqrt(np.mean(all_errors ** 2)):.2f} deg, "
            f"MAE={np.mean(all_errors):.2f} deg"
        )
    print(f"Saved: {pred_csv}")
    print(f"Saved: {mse_csv}")
    print(f"Saved: {mse_plot_path}")
    print(f"Saved: {doa_plot_path}")
    for path in mean_error_bar_paths:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
