from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


RESULT_DIR = Path(__file__).resolve().parent / "evaluation_results"
PICTURE_DIR = Path(__file__).resolve().parent / "evaluation_pictures"

VOICE_EVALUATIONS = {
    "session_20260521_005414_recordings_voice": RESULT_DIR
    / "session_20260521_005414_recordings_voice_srp_phat_mse_by_angle.csv",
    "session_20260517_161547_recordings": RESULT_DIR
    / "session_20260517_161547_recordings_srp_phat_mse_by_angle.csv",
}

PAIR_LABELS_HU = {
    "2_pairs_opposite": "2 mikrofonpár - szemközti párok",
    "4_pairs_perimeter": "4 mikrofonpár - szomszédos párok",
}

COLORS = {
    "2_pairs_opposite": "#4477AA",
    "4_pairs_perimeter": "#EE6677",
    "6_pairs_all": "#228833",
}


MIN_CYCLES_PER_360 = 4.0
MAX_CYCLES_PER_360 = 6.0
SPIKE_THRESHOLD_DEG = 3.0


def fit_model(t, amplitude, omega, phase, offset):
    return offset + amplitude * np.sin(omega * t + phase)


def initial_guess(t, y):
    amplitude = max((np.max(y) - np.min(y)) / 2.0, 0.1)
    offset = max(np.mean(y), amplitude)
    omega = 2.0 * np.pi * 4.0 / 360.0
    return [amplitude, omega, 0.0, offset]


def drop_spiky_points(t, y, threshold=SPIKE_THRESHOLD_DEG):
    keep = np.ones(len(y), dtype=bool)
    if len(y) < 3:
        return t, y, keep

    for idx in range(1, len(y) - 1):
        left_jump = abs(y[idx] - y[idx - 1])
        right_jump = abs(y[idx] - y[idx + 1])
        neighbor_jump = abs(y[idx - 1] - y[idx + 1])

        # Drop isolated points that jump away from both neighbours.
        if left_jump > threshold and right_jump > threshold and neighbor_jump <= threshold:
            keep[idx] = False

    return t[keep], y[keep], keep


def fit_curve(t, y):
    t_fit, y_fit, keep_mask = drop_spiky_points(t, y)
    base = initial_guess(t_fit, y_fit)
    guesses = [
        base,
        [base[0], 2.0 * np.pi * 5.0 / 360.0, 0.0, base[3]],
        [base[0], 2.0 * np.pi * 6.0 / 360.0, 0.0, base[3]],
        [base[0], 2.0 * np.pi * 4.0 / 360.0, np.pi, base[3]],
        [base[0], 2.0 * np.pi * 5.0 / 360.0, np.pi, base[3]],
    ]
    min_omega = 2.0 * np.pi * MIN_CYCLES_PER_360 / 360.0
    max_omega = 2.0 * np.pi * MAX_CYCLES_PER_360 / 360.0
    bounds = (
        [0.0, min_omega, -4.0 * np.pi, 0.0],
        [200.0, max_omega, 4.0 * np.pi, 200.0],
    )

    best_params = None
    best_rmse = np.inf
    for guess in guesses:
        try:
            params, _ = curve_fit(
                fit_model,
                t_fit,
                y_fit,
                p0=guess,
                bounds=bounds,
                maxfev=20000,
            )
        except RuntimeError:
            continue
        if params[3] < params[0]:
            continue
        residual = fit_model(t_fit, *params) - y_fit
        rmse = float(np.sqrt(np.mean(residual ** 2)))
        if rmse < best_rmse:
            best_rmse = rmse
            best_params = params

    if best_params is None:
        raise RuntimeError("Curve fit failed for all initial guesses.")

    return best_params, best_rmse, keep_mask


def plot_fit(dataset_name, scenario, rows, params, rmse, keep_mask):
    t = rows["truth_deg"].to_numpy(dtype=float)
    y = rows["mae_deg"].to_numpy(dtype=float)
    dense_t = np.linspace(t.min(), t.max(), 1000)
    dense_y = fit_model(dense_t, *params)

    amplitude, omega, phase, offset = params
    period = 2.0 * np.pi / omega if omega > 0 else np.inf
    y_max = max(np.max(y), np.max(dense_y)) + 1.0

    path = PICTURE_DIR / f"{dataset_name}_{scenario}_atlagos_hiba_illesztett_gorbe.png"
    plt.figure(figsize=(16, 5))
    plt.bar(t[keep_mask], y[keep_mask], width=3.5, color=COLORS[scenario], alpha=0.45, label="Illesztéshez használt mért hiba")
    if np.any(~keep_mask):
        plt.scatter(t[~keep_mask], y[~keep_mask], color="red", marker="x", s=55, label="Eldobott kiugró pont")
    plt.plot(dense_t, dense_y, color="black", linewidth=2.2, label="Illesztett görbe")
    plt.xlabel("Valós beesési szög (fok)")
    plt.ylabel("Átlagos abszolút hiba (fok)")
    plt.title(f"SRP-PHAT hiba görbeillesztés - {PAIR_LABELS_HU[scenario]}")
    plt.text(
        0.01,
        0.96,
        (
            "Illesztett alak: C + A sin(ωt + φ), C ≥ A\n"
            f"A={amplitude:.3f}, ω={omega:.5f}, φ={phase:.3f}, "
            f"C={offset:.3f}, periódus={period:.1f} fok, RMSE={rmse:.3f}"
        ),
        transform=plt.gca().transAxes,
        va="top",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )
    plt.xticks(np.arange(0, 360, 10), rotation=90)
    plt.ylim(0, max(5.0, y_max))
    plt.grid(axis="y", alpha=0.3)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def main():
    PICTURE_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    fit_rows = []
    written = []
    for dataset_name, csv_path in VOICE_EVALUATIONS.items():
        if not csv_path.exists():
            print(f"Missing CSV, skipping: {csv_path}")
            continue

        data = pd.read_csv(csv_path)
        for scenario in PAIR_LABELS_HU:
            rows = data[data["scenario"] == scenario].sort_values("truth_deg")
            if rows.empty:
                continue

            t = rows["truth_deg"].to_numpy(dtype=float)
            y = rows["mae_deg"].to_numpy(dtype=float)
            params, rmse, keep_mask = fit_curve(t, y)
            path = plot_fit(dataset_name, scenario, rows, params, rmse, keep_mask)
            written.append(path)

            amplitude, omega, phase, offset = params
            fit_rows.append({
                "dataset": dataset_name,
                "scenario": scenario,
                "A": amplitude,
                "omega": omega,
                "phi": phase,
                "C": offset,
                "period_deg": 2.0 * np.pi / omega if omega > 0 else np.inf,
                "fit_rmse_deg": rmse,
                "dropped_points": int(np.sum(~keep_mask)),
            })

    fit_csv = RESULT_DIR / "voice_error_curve_fit_parameters.csv"
    pd.DataFrame(fit_rows).to_csv(fit_csv, index=False)
    print(f"Saved: {fit_csv}")
    for path in written:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
