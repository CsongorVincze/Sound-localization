from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PICTURE_DIR = Path(__file__).resolve().parent / "evaluation_pictures"
RESULT_DIR = Path(__file__).resolve().parent / "evaluation_results"
VOICE_RESULT_CSV = (
    RESULT_DIR
    / "session_20260521_005414_recordings_voice_srp_phat_mse_by_angle.csv"
)

SPEED_OF_SOUND = 343.0
MIC_RADIUS_M = 0.035

PAIR_SCENARIOS = {
    "2_pairs_opposite": [(0, 2), (1, 3)],
    "4_pairs_perimeter": [(0, 1), (1, 2), (2, 3), (3, 0)],
    "6_pairs_all": [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
}

INDIVIDUAL_PAIRS = {
    "pair_0_1": [(0, 1)],
    "pair_0_2": [(0, 2)],
    "pair_0_3": [(0, 3)],
    "pair_1_2": [(1, 2)],
    "pair_1_3": [(1, 3)],
    "pair_2_3": [(2, 3)],
}

SCENARIO_LABELS_HU = {
    "2_pairs_opposite": "2 mikrofonpár - szemközti párok",
    "4_pairs_perimeter": "4 mikrofonpár - szomszédos párok",
    "6_pairs_all": "6 mikrofonpár - összes pár",
}

COLORS = {
    "2_pairs_opposite": "#4477AA",
    "4_pairs_perimeter": "#EE6677",
    "6_pairs_all": "#228833",
}


def respeaker_mic_positions(radius_m=MIC_RADIUS_M):
    mic_angles = np.deg2rad([-45.0, 45.0, 135.0, 225.0])
    return np.column_stack((
        radius_m * np.cos(mic_angles),
        radius_m * np.sin(mic_angles),
    ))


def dummy_crlb_shape(angles_deg, pairs, mic_positions):
    """
    Geometry-only CRLB shape for far-field 2D DoA from TDOA measurements.

    tau_ij(theta) = dot(mic_i - mic_j, u(theta)) / c
    d tau_ij / d theta = dot(mic_i - mic_j, u_perp(theta)) / c

    Ignoring SNR, bandwidth, and delay-estimator variance means the absolute
    scale is unknown. The useful part here is the angle-dependent shape:

        sigma_theta(theta) proportional to 1 / sqrt(sum_pair sensitivity^2)
    """
    theta = np.deg2rad(angles_deg)
    u_perp = np.column_stack((-np.sin(theta), np.cos(theta)))
    fisher_shape = np.zeros_like(theta, dtype=np.float64)

    for i, j in pairs:
        baseline = mic_positions[i] - mic_positions[j]
        delay_sensitivity = u_perp @ baseline / SPEED_OF_SOUND
        fisher_shape += delay_sensitivity ** 2

    relative_std_rad = 1.0 / np.sqrt(np.maximum(fisher_shape, 1e-30))
    relative_std_deg = np.rad2deg(relative_std_rad)
    return relative_std_deg / np.mean(relative_std_deg)


def build_dummy_table():
    angles = np.arange(0.0, 360.0, 5.0)
    mic_positions = respeaker_mic_positions()
    rows = []

    for scenario, pairs in PAIR_SCENARIOS.items():
        rel_error = dummy_crlb_shape(angles, pairs, mic_positions)
        for angle, value in zip(angles, rel_error):
            rows.append({
                "scenario": scenario,
                "truth_deg": angle,
                "relative_theoretical_error": float(value),
            })

    return pd.DataFrame(rows)


def plot_relative_crlb(crlb_df, path):
    plt.figure(figsize=(14, 5))
    for scenario in PAIR_SCENARIOS:
        data = crlb_df[crlb_df["scenario"] == scenario]
        plt.plot(
            data["truth_deg"],
            data["relative_theoretical_error"],
            color=COLORS[scenario],
            linewidth=2,
            label=SCENARIO_LABELS_HU[scenario],
        )

    plt.xlabel("Valós beesési szög (fok)")
    plt.ylabel("Relatív elméleti hiba")
    plt.title("Geometriai CRLB jellegű relatív DoA hiba SNR nélkül")
    plt.xticks(np.arange(0, 360, 20))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_scaled_vs_measured(crlb_df, measured_csv, path):
    if not measured_csv.exists():
        print(f"Measured CSV not found, skipping comparison: {measured_csv}")
        return False

    measured = pd.read_csv(measured_csv)
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)

    for ax, scenario in zip(axes, PAIR_SCENARIOS):
        theory = crlb_df[crlb_df["scenario"] == scenario].copy()
        meas = measured[measured["scenario"] == scenario].copy()
        theory = theory.sort_values("truth_deg")
        meas = meas.sort_values("truth_deg")

        # SNR is intentionally omitted, so the CRLB has no absolute scale.
        # Scale it to the measured mean error to compare only the angle trend.
        scaled_theory = (
            theory["relative_theoretical_error"].to_numpy()
            * meas["mae_deg"].mean()
        )

        ax.plot(
            theory["truth_deg"],
            scaled_theory,
            color=COLORS[scenario],
            linewidth=2,
            label="Geometriai CRLB alak, mérési átlaghoz skálázva",
        )
        ax.bar(
            meas["truth_deg"],
            meas["mae_deg"],
            width=3.5,
            color=COLORS[scenario],
            alpha=0.35,
            label="Mért átlagos abszolút hiba",
        )
        ax.set_ylabel("Hiba (fok)")
        ax.set_title(SCENARIO_LABELS_HU[scenario])
        ax.grid(axis="y", alpha=0.3)
        ax.legend(loc="upper left")

    axes[-1].set_xlabel("Valós beesési szög (fok)")
    axes[-1].set_xticks(np.arange(0, 360, 20))
    fig.suptitle("Mért SRP-PHAT hiba és SNR nélküli geometriai CRLB alak összehasonlítása")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def plot_individual_pair_crlb(path):
    angles = np.arange(0.0, 360.0, 1.0)
    mic_positions = respeaker_mic_positions()

    plt.figure(figsize=(14, 5))
    for label, pairs in INDIVIDUAL_PAIRS.items():
        rel_error = dummy_crlb_shape(angles, pairs, mic_positions)
        rel_error = np.clip(rel_error / np.median(rel_error), 0, 8)
        plt.plot(angles, rel_error, linewidth=1.5, label=label.replace("_", " "))

    plt.xlabel("Valós beesési szög (fok)")
    plt.ylabel("Relatív elméleti hiba, mediánra normálva")
    plt.title("Egyedi mikrofonpárok geometriai DoA hibaalakja SNR nélkül")
    plt.xticks(np.arange(0, 360, 20))
    plt.ylim(0, 8)
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    PICTURE_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    crlb_df = build_dummy_table()
    csv_path = RESULT_DIR / "dummy_geometry_crlb_by_angle.csv"
    relative_plot = PICTURE_DIR / "dummy_geometry_crlb_relative_error.png"
    comparison_plot = PICTURE_DIR / "dummy_geometry_crlb_vs_measured_voice.png"
    individual_pair_plot = PICTURE_DIR / "dummy_geometry_crlb_individual_pairs.png"

    crlb_df.to_csv(csv_path, index=False)
    plot_relative_crlb(crlb_df, relative_plot)
    comparison_written = plot_scaled_vs_measured(
        crlb_df,
        VOICE_RESULT_CSV,
        comparison_plot,
    )
    plot_individual_pair_crlb(individual_pair_plot)

    print(f"Saved: {csv_path}")
    print(f"Saved: {relative_plot}")
    if comparison_written:
        print(f"Saved: {comparison_plot}")
    print(f"Saved: {individual_pair_plot}")


if __name__ == "__main__":
    main()
