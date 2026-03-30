"""
optimize_lif_real.py — Parameter Optimization on Real Audio

Takes a single 3-second recorded .npy file (with known true angle) and runs
a grid search over tau_leaky and v_thresh to find the parameters that produce
the most accurate LIF neuron firing.

Usage:
    python optimize_lif_real.py recordings/Goldberg_3sec/Goldberg/angle_045_servo_090.npy
    python optimize_lif_real.py recordings/Goldberg_3sec/Goldberg/angle_045_servo_090.npy --true_angle 45

If true_angle is not given, it is parsed from the filename (angle_XXX_...).
"""
import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

MEASURE_DIR = Path(__file__).parent
PROJECT_DIR = MEASURE_DIR.parent
sys.path.insert(0, str(MEASURE_DIR))
sys.path.insert(0, str(PROJECT_DIR / "Brian_2_sim"))

import warnings
warnings.filterwarnings('ignore')

from brian2 import prefs
prefs.codegen.target = 'numpy'

# =============================================================================
# CONFIGURATION
# =============================================================================
SAMPLE_RATE = 16000
LIF_RESOLUTION_DEG = 15

# Grid search ranges
TAU_VALUES = np.linspace(30, 150, 13)    # 30 to 150 us, 13 steps (~10us apart)
VTHRESH_VALUES = np.linspace(0.3, 1.3, 11)  # 0.3 to 1.3 V, 11 steps (0.1V apart)

# =============================================================================
# LIF EVALUATION (single recording, single param set)
# =============================================================================

def run_lif_single(mic_audio, sample_rate, duration_sec, tau_leaky_us, v_thresh_V):
    """
    Run the Brian2 LIF on one recording with given parameters.

    Returns:
        best_angle: the most-fired neuron angle (degrees), or None
        n_fired: total number of neuron spikes
        all_fired_angles: list of all fired neuron angles
    """
    from scipy.signal import find_peaks, resample
    from brian2 import (start_scope, defaultclock, us, ms, second, meter,
                        volt, SpikeGeneratorGroup, SpikeMonitor, run as brian_run)
    from kiindulo_kod import create_network, get_respeaker_array_geometry, get_target_angles

    start_scope()
    defaultclock.dt = 1 * us

    mic_x, mic_y, num_mics = get_respeaker_array_geometry()
    target_angles, deg = get_target_angles(LIF_RESOLUTION_DEG)
    c_sound = 343 * meter / second

    # Spike extraction
    UPSAMPLE_FACTOR = 16
    HIGH_FS = sample_rate * UPSAMPLE_FACTOR
    mics_audio_up = resample(mic_audio, len(mic_audio) * UPSAMPLE_FACTOR, axis=0)

    global_max = np.max(np.abs(mics_audio_up))
    if global_max < 1e-10:
        return None, 0, []
    threshold = 0.20 * global_max

    snn_to_ch_map = {0: 0, 1: 3, 2: 2, 3: 1}
    all_raw_spikes = []
    for snn_idx in range(4):
        ch_idx = snn_to_ch_map[snn_idx]
        audio_ch = mics_audio_up[:, ch_idx]
        min_dist = int(HIGH_FS * 0.001)
        peaks, _ = find_peaks(np.abs(audio_ch), height=threshold, distance=min_dist)
        for t in peaks / HIGH_FS:
            all_raw_spikes.append((t, snn_idx))

    all_raw_spikes.sort(key=lambda x: x[0])

    all_indices = []
    all_times = []
    current_cluster = []

    def process_cluster(cluster):
        if not cluster:
            return
        for t, idx in cluster:
            all_times.append(t)
            all_indices.append(idx)

    for spike in all_raw_spikes:
        if not current_cluster:
            current_cluster.append(spike)
        else:
            if spike[0] - current_cluster[-1][0] < 0.050:
                current_cluster.append(spike)
            else:
                process_cluster(current_cluster)
                current_cluster = [spike]
    if current_cluster:
        process_cluster(current_cluster)

    if not all_times:
        return None, 0, []

    min_t = np.min(all_times)
    if min_t < 0:
        all_times = [t - min_t for t in all_times]

    mics_sg = SpikeGeneratorGroup(num_mics, all_indices,
                                   np.array(all_times) * second)

    mott_neurons, synapses, wta_syns = create_network(
        mics=mics_sg,
        mic_x=mic_x, mic_y=mic_y,
        target_angles=target_angles,
        tau_leaky=tau_leaky_us * us,
        v_thresh=v_thresh_V * volt,
        c_sound=c_sound
    )

    spike_mon = SpikeMonitor(mott_neurons)
    brian_run(duration_sec * second + 50 * ms, report='text')

    fired_angles = []
    for spike_idx in spike_mon.i:
        fired_angles.append(float(target_angles[int(spike_idx)] / deg))

    if fired_angles:
        unique, counts = np.unique(fired_angles, return_counts=True)
        best_angle = unique[np.argmax(counts)]
        return best_angle, len(fired_angles), fired_angles
    else:
        return None, 0, []


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python optimize_lif_real.py <path_to_angle_XXX.npy> [--true_angle N]")
        print("\nExample:")
        print("  python optimize_lif_real.py recordings/Goldberg_3sec/Goldberg/angle_045_servo_090.npy")
        sys.exit(1)

    npy_path = Path(sys.argv[1])
    if not npy_path.exists():
        print(f"ERROR: File not found: {npy_path}")
        sys.exit(1)

    # Parse true angle
    true_angle = None
    if '--true_angle' in sys.argv:
        idx = sys.argv.index('--true_angle')
        true_angle = int(sys.argv[idx + 1])
    else:
        # Try to parse from filename: angle_045_servo_090.npy
        try:
            parts = npy_path.stem.split("_")
            true_angle = int(parts[1])
        except (IndexError, ValueError):
            true_angle = int(input("Could not parse angle from filename. Enter true angle (degrees): "))

    print("=" * 60)
    print(" LIF Parameter Optimization on Real Audio")
    print("=" * 60)
    print(f"    File: {npy_path}")
    print(f"    True angle: {true_angle}°")
    print(f"    Grid: {len(TAU_VALUES)} tau × {len(VTHRESH_VALUES)} v_thresh = "
          f"{len(TAU_VALUES) * len(VTHRESH_VALUES)} simulations")
    print(f"    tau_leaky:  {TAU_VALUES[0]:.0f} – {TAU_VALUES[-1]:.0f} us")
    print(f"    v_thresh:   {VTHRESH_VALUES[0]:.2f} – {VTHRESH_VALUES[-1]:.2f} V")

    # Load audio
    recording = np.load(npy_path)
    mic_audio = recording[:, 1:5].astype(np.float64)
    duration_sec = len(mic_audio) / SAMPLE_RATE
    print(f"    Duration: {duration_sec:.1f}s")

    # Grid search
    total_runs = len(TAU_VALUES) * len(VTHRESH_VALUES)
    error_matrix = np.full((len(VTHRESH_VALUES), len(TAU_VALUES)), np.nan)
    success_matrix = np.zeros((len(VTHRESH_VALUES), len(TAU_VALUES)))
    predicted_matrix = np.full((len(VTHRESH_VALUES), len(TAU_VALUES)), np.nan)
    nfired_matrix = np.zeros((len(VTHRESH_VALUES), len(TAU_VALUES)))

    best_error = float('inf')
    best_params = (None, None)
    best_predicted = None

    run_count = 0
    start_time = time.time()

    print(f"\n    Starting grid search ({total_runs} simulations)...\n")

    for i, vt in enumerate(VTHRESH_VALUES):
        for j, tl in enumerate(TAU_VALUES):
            run_count += 1
            t0 = time.time()

            pred, n_fired, _ = run_lif_single(
                mic_audio, SAMPLE_RATE, duration_sec, tl, vt
            )

            elapsed = time.time() - t0
            total_elapsed = time.time() - start_time
            eta = (total_elapsed / run_count) * (total_runs - run_count)

            nfired_matrix[i, j] = n_fired

            if pred is not None:
                diff = abs(true_angle - pred)
                if diff > 180:
                    diff = 360 - diff

                error_matrix[i, j] = diff
                success_matrix[i, j] = 1
                predicted_matrix[i, j] = pred

                status = f"→ {pred:5.0f}° (err={diff:4.0f}°)"

                if diff < best_error:
                    best_error = diff
                    best_params = (vt, tl)
                    best_predicted = pred
            else:
                error_matrix[i, j] = 90  # penalty
                success_matrix[i, j] = 0
                status = "→ NO FIRE"

            print(f"    [{run_count:3d}/{total_runs}] "
                  f"τ={tl:5.0f}us Vt={vt:.2f}V  {status}  "
                  f"({elapsed:.1f}s, ETA {eta/60:.0f}min)")

    total_time = time.time() - start_time

    # --- Direction auto-detection ---
    # Check if (360-predicted) consistently gives lower error than raw predicted
    valid_preds = predicted_matrix[~np.isnan(predicted_matrix)]
    if len(valid_preds) > 0:
        err_normal = np.abs(true_angle - valid_preds)
        err_normal = np.where(err_normal > 180, 360 - err_normal, err_normal)

        flipped_preds = (360 - valid_preds) % 360
        err_flipped = np.abs(true_angle - flipped_preds)
        err_flipped = np.where(err_flipped > 180, 360 - err_flipped, err_flipped)

        if np.mean(err_flipped) < np.mean(err_normal) * 0.7:
            print(f"\n  ⚠ DIRECTION MISMATCH DETECTED!")
            print(f"    Mean normal error:  {np.mean(err_normal):.1f}°")
            print(f"    Mean flipped error: {np.mean(err_flipped):.1f}°")
            print(f"    → Recomputing with flipped angles: (360 - predicted) % 360")

            # Recompute everything with flipped predictions
            for i in range(len(VTHRESH_VALUES)):
                for j in range(len(TAU_VALUES)):
                    if not np.isnan(predicted_matrix[i, j]):
                        flipped = (360 - predicted_matrix[i, j]) % 360
                        predicted_matrix[i, j] = flipped
                        diff = abs(true_angle - flipped)
                        if diff > 180:
                            diff = 360 - diff
                        error_matrix[i, j] = diff

            # Recalculate best
            best_error = float('inf')
            for i in range(len(VTHRESH_VALUES)):
                for j in range(len(TAU_VALUES)):
                    if not np.isnan(error_matrix[i, j]) and error_matrix[i, j] < best_error:
                        best_error = error_matrix[i, j]
                        best_params = (VTHRESH_VALUES[i], TAU_VALUES[j])
                        best_predicted = predicted_matrix[i, j]
        else:
            print(f"\n  ✓ Direction OK (normal err={np.mean(err_normal):.1f}°, "
                  f"flipped err={np.mean(err_flipped):.1f}°)")

    # Results
    print(f"\n{'='*60}")
    print(f" OPTIMIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"    Total time: {total_time/60:.1f} min")
    print(f"    True angle: {true_angle}°")
    print(f"    Best v_thresh:  {best_params[0]:.2f} V")
    print(f"    Best tau_leaky: {best_params[1]:.0f} us")
    print(f"    Predicted:      {best_predicted}°")
    print(f"    Error:          {best_error:.0f}°")

    # Save results
    out_dir = npy_path.parent
    result_name = f"optimize_{npy_path.stem}"

    # Save to text
    txt_path = out_dir / f"{result_name}.txt"
    with open(txt_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("LIF PARAMETER OPTIMIZATION — REAL AUDIO\n")
        f.write("=" * 60 + "\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"File: {npy_path}\n")
        f.write(f"True angle: {true_angle}°\n")
        f.write(f"Duration: {duration_sec:.1f}s\n")
        f.write(f"Grid: {len(TAU_VALUES)} tau x {len(VTHRESH_VALUES)} v_thresh\n\n")
        f.write(f"OPTIMAL PARAMETERS:\n")
        f.write(f"  v_thresh:  {best_params[0]:.2f} V\n")
        f.write(f"  tau_leaky: {best_params[1]:.0f} us\n")
        f.write(f"  Predicted: {best_predicted}°\n")
        f.write(f"  Error:     {best_error:.0f}°\n\n")

        f.write("TAU_LEAKY sweep (us):\n")
        f.write(", ".join([f"{t:.0f}" for t in TAU_VALUES]) + "\n\n")
        f.write("V_THRESH sweep (V):\n")
        f.write(", ".join([f"{v:.2f}" for v in VTHRESH_VALUES]) + "\n\n")

        f.write("ERROR MATRIX (rows=v_thresh, cols=tau_leaky):\n")
        np.savetxt(f, error_matrix, fmt="%6.1f", delimiter=", ")
        f.write("\n")

        f.write("SUCCESS MATRIX (1=fired, 0=silent):\n")
        np.savetxt(f, success_matrix, fmt="%4.0f", delimiter=", ")
        f.write("\n")

        f.write("PREDICTED ANGLE MATRIX:\n")
        np.savetxt(f, predicted_matrix, fmt="%6.0f", delimiter=", ")
        f.write("\n")

        f.write("NUM FIRES MATRIX:\n")
        np.savetxt(f, nfired_matrix, fmt="%6.0f", delimiter=", ")
    print(f"    Saved: {txt_path}")

    # Save raw data
    npy_out = out_dir / f"{result_name}.npy"
    np.save(npy_out, {
        'true_angle': true_angle,
        'tau_values': TAU_VALUES,
        'vthresh_values': VTHRESH_VALUES,
        'error_matrix': error_matrix,
        'success_matrix': success_matrix,
        'predicted_matrix': predicted_matrix,
        'nfired_matrix': nfired_matrix,
        'best_params': best_params,
        'best_error': best_error,
        'best_predicted': best_predicted,
        'source_file': str(npy_path),
        'timestamp': datetime.now().isoformat(),
    })
    print(f"    Saved: {npy_out}")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(f"LIF Parameter Optimization — True angle: {true_angle}°\n"
                 f"Best: τ={best_params[1]:.0f}us, Vt={best_params[0]:.2f}V → "
                 f"pred={best_predicted}° (err={best_error:.0f}°)",
                 fontsize=13, fontweight='bold')

    extent = [TAU_VALUES[0], TAU_VALUES[-1], VTHRESH_VALUES[0], VTHRESH_VALUES[-1]]

    # 1. Error heatmap
    ax1 = axes[0, 0]
    err_display = np.where(np.isnan(error_matrix), 90, error_matrix)
    im1 = ax1.imshow(err_display, origin='lower', aspect='auto', cmap='viridis_r',
                     extent=extent, vmin=0, vmax=max(15, np.nanmin(error_matrix) + 20))
    plt.colorbar(im1, ax=ax1, label='Error (°)')
    ax1.plot(best_params[1], best_params[0], 'r*', markersize=18, label='Best')
    ax1.set_xlabel('τ_leaky (μs)')
    ax1.set_ylabel('V_thresh (V)')
    ax1.set_title('Localization Error\n(lighter = better)')
    ax1.legend()

    # 2. Success heatmap
    ax2 = axes[0, 1]
    im2 = ax2.imshow(success_matrix, origin='lower', aspect='auto', cmap='plasma',
                     extent=extent)
    plt.colorbar(im2, ax=ax2, label='Fired (1=yes)')
    ax2.plot(best_params[1], best_params[0], 'w*', markersize=18, label='Best')
    ax2.set_xlabel('τ_leaky (μs)')
    ax2.set_ylabel('V_thresh (V)')
    ax2.set_title('Firing Success\n(yellow = fired)')
    ax2.legend()

    # 3. Predicted angle heatmap
    ax3 = axes[1, 0]
    im3 = ax3.imshow(predicted_matrix, origin='lower', aspect='auto', cmap='hsv',
                     extent=extent, vmin=0, vmax=360)
    plt.colorbar(im3, ax=ax3, label='Predicted Angle (°)')
    ax3.plot(best_params[1], best_params[0], 'k*', markersize=18, label='Best')
    ax3.set_xlabel('τ_leaky (μs)')
    ax3.set_ylabel('V_thresh (V)')
    ax3.set_title(f'Predicted Angle\n(true = {true_angle}°)')
    ax3.legend()

    # 4. Number of fires heatmap
    ax4 = axes[1, 1]
    im4 = ax4.imshow(nfired_matrix, origin='lower', aspect='auto', cmap='hot',
                     extent=extent)
    plt.colorbar(im4, ax=ax4, label='# Neuron Spikes')
    ax4.plot(best_params[1], best_params[0], 'c*', markersize=18, label='Best')
    ax4.set_xlabel('τ_leaky (μs)')
    ax4.set_ylabel('V_thresh (V)')
    ax4.set_title('Total Neuron Fires')
    ax4.legend()

    plt.tight_layout()
    plot_path = out_dir / f"{result_name}.png"
    plt.savefig(plot_path, dpi=150)
    print(f"    Saved: {plot_path}")
    plt.show()

    print("\nDone.")
