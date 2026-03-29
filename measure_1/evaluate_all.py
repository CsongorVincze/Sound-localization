"""
evaluate_all.py — Offline DoA Algorithm Comparison

Loads pre-recorded audio sweeps (from record_sweep.py) and evaluates them
with all available DoA algorithms:

  1. GCC-PHAT
  2. SRP-PHAT
  3. Basic CC
  4. LIF SNN (Brian2 coincidence detection)

Outputs:
  - A comparison plot (PNG) with tracking, error bars, box plot, and summary
  - A text file (lif_neuron_firings.txt) logging which neuron fired at each angle

Usage:
    python evaluate_all.py                          # picks the latest recording folder
    python evaluate_all.py recordings/sweep_20260330_001500  # specific folder
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directories so we can import from both measure_1 and Brian_2_sim
MEASURE_DIR = Path(__file__).parent
PROJECT_DIR = MEASURE_DIR.parent
sys.path.insert(0, str(MEASURE_DIR))
sys.path.insert(0, str(PROJECT_DIR / "Brian_2_sim"))

from my_algos import get_gcc_phat_angle, get_srp_phat_angle, get_basic_cc_angle

# =============================================================================
# CONFIGURATION
# =============================================================================
SAMPLE_RATE = 16000

# Classical algorithms
ALGORITHMS = [
    ("GCC-PHAT", get_gcc_phat_angle, "#00d2ff"),
    ("SRP-PHAT", get_srp_phat_angle, "#4ecdc4"),
    ("Basic CC", get_basic_cc_angle, "#ff6b6b"),
]

# LIF SNN configuration
LIF_RESOLUTION_DEG = 15   # angular resolution of the LIF neuron array
LIF_COLOR = "#9b59b6"

# =============================================================================
# FIND RECORDING FOLDER
# =============================================================================

def find_recording_folder(arg=None):
    """Find the recording folder from command line arg or pick the latest."""
    recordings_dir = MEASURE_DIR / "recordings"

    if arg:
        p = Path(arg)
        if p.is_dir():
            return p
        p = recordings_dir / arg
        if p.is_dir():
            return p
        print(f"ERROR: folder not found: {arg}")
        sys.exit(1)

    if not recordings_dir.is_dir():
        print("ERROR: No 'recordings' folder found. Run record_sweep.py first.")
        sys.exit(1)

    folders = sorted([f for f in recordings_dir.iterdir() if f.is_dir()])
    if not folders:
        print("ERROR: No recording sessions found in recordings/")
        sys.exit(1)

    return folders[-1]  # latest


# =============================================================================
# LIF SNN EVALUATION
# =============================================================================

def evaluate_lif_snn(mic_audio, sample_rate, duration_sec):
    """
    Run the Brian2 LIF coincidence detection on a single recording.

    Returns:
        fired_angles: list of angles (in degrees) of neurons that fired
        all_fired: list of (time_sec, angle_deg) for every spike
    """
    try:
        from scipy.signal import find_peaks, resample
        from brian2 import (start_scope, defaultclock, us, ms, second, meter,
                            volt, mV, SpikeGeneratorGroup, SpikeMonitor,
                            StateMonitor, run as brian_run, array, cos, sin,
                            max, min)
        from kiindulo_kod import create_network, get_respeaker_array_geometry, get_target_angles
    except ImportError as e:
        print(f"    LIF SNN skipped (missing dependency: {e})")
        return [], []

    start_scope()
    defaultclock.dt = 1 * us

    mic_x, mic_y, num_mics = get_respeaker_array_geometry()
    target_angles, deg = get_target_angles(LIF_RESOLUTION_DEG)
    num_neurons = len(target_angles)
    c_sound = 343 * meter / second
    respeaker_tau_leaky = 55 * us

    # --- Spike extraction (same approach as real_audio_sim.py) ---
    UPSAMPLE_FACTOR = 16
    HIGH_FS = sample_rate * UPSAMPLE_FACTOR

    mics_audio_up = resample(mic_audio, len(mic_audio) * UPSAMPLE_FACTOR, axis=0)

    global_max = np.max(np.abs(mics_audio_up))
    if global_max < 1e-10:
        return [], []
    threshold = 0.20 * global_max

    snn_to_ch_map = {0: 0, 1: 3, 2: 2, 3: 1}

    all_raw_spikes = []
    for snn_idx in range(4):
        ch_idx = snn_to_ch_map[snn_idx]
        audio_ch = mics_audio_up[:, ch_idx]

        min_dist = int(HIGH_FS * 0.001)
        peaks, _ = find_peaks(np.abs(audio_ch), height=threshold, distance=min_dist)
        peak_times_sec = peaks / HIGH_FS

        for t in peak_times_sec:
            all_raw_spikes.append((t, snn_idx))

    all_raw_spikes.sort(key=lambda x: x[0])

    # Clustering
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
        return [], []

    # Shift times to start from 0
    min_t = np.min(all_times)
    if min_t < 0:
        all_times = [t - min_t for t in all_times]

    mics_sg = SpikeGeneratorGroup(num_mics, all_indices, np.array(all_times) * second)

    mott_neurons, synapses, wta_syns = create_network(
        mics=mics_sg,
        mic_x=mic_x,
        mic_y=mic_y,
        target_angles=target_angles,
        tau_leaky=respeaker_tau_leaky,
        c_sound=c_sound
    )

    spike_mon = SpikeMonitor(mott_neurons)
    brian_run(duration_sec * second + 50 * ms, report=None)

    # Collect results
    fired_angles = []
    all_fired = []
    for spike_idx, spike_t in zip(spike_mon.i, spike_mon.t):
        angle_deg = float(target_angles[int(spike_idx)] / deg)
        fired_angles.append(angle_deg)
        all_fired.append((float(spike_t / second), angle_deg))

    return fired_angles, all_fired


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    folder = find_recording_folder(arg)

    print("=" * 60)
    print(" Offline DoA Algorithm Evaluation")
    print("=" * 60)
    print(f"    Folder: {folder}")

    # Load metadata
    meta_path = folder / "metadata.npy"
    if meta_path.exists():
        meta = np.load(meta_path, allow_pickle=True).item()
        SAMPLE_RATE = meta.get('sample_rate', SAMPLE_RATE)
        step_inc = meta.get('step_increment', 5)
        rec_dur = meta.get('recording_duration', 3.0)
        alignment_offset = meta.get('alignment_offset', 45)
        print(f"    Sample rate: {SAMPLE_RATE} Hz")
        print(f"    Step: {step_inc}°, Duration: {rec_dur}s")
        print(f"    Alignment offset: {alignment_offset}°")
    else:
        step_inc = 5
        rec_dur = 3.0
        alignment_offset = 45

    # Find all recording files
    npy_files = sorted(folder.glob("angle_*.npy"))
    if not npy_files:
        print("ERROR: No angle_*.npy files found!")
        sys.exit(1)

    print(f"    Found {len(npy_files)} recordings")

    # Parse angles from filenames
    recordings = []
    for f in npy_files:
        parts = f.stem.split("_")
        # angle_000_servo_045.npy
        true_angle = int(parts[1])
        servo_pos = int(parts[3])
        recordings.append((true_angle, servo_pos, f))

    # --- Calibration offset detection (using first recording = 0° position) ---
    print("\n    Computing calibration offsets from first recording...")
    first_rec = np.load(recordings[0][2])
    first_audio = first_rec[:, 1:5].astype(np.float64)

    calibration_offsets = {}
    for name, algo_func, _ in ALGORITHMS:
        try:
            calibration_offsets[name] = algo_func(first_audio, SAMPLE_RATE)
        except Exception:
            calibration_offsets[name] = 0.0

    # Direction detection: compare first and second recording
    direction_sign = 1
    if len(recordings) >= 2:
        second_rec = np.load(recordings[1][2])
        second_audio = second_rec[:, 1:5].astype(np.float64)
        votes = []
        for name, algo_func, _ in ALGORITHMS:
            try:
                a1 = calibration_offsets[name]
                a2 = algo_func(second_audio, SAMPLE_RATE)
                diff = a2 - a1
                if diff > 180:
                    diff -= 360
                if diff < -180:
                    diff += 360
                votes.append(np.sign(diff))
            except Exception:
                pass
        if votes:
            direction_sign = 1 if np.mean(votes) >= 0 else -1

    print(f"    Direction: {'normal' if direction_sign == 1 else 'REVERSED (auto-corrected)'}")
    for name, _, _ in ALGORITHMS:
        print(f"      {name}: cal_offset = {calibration_offsets.get(name, 0):.1f}°")

    # --- Evaluate all recordings ---
    print("\n" + "=" * 60)
    print(" Evaluating Algorithms")
    print("=" * 60)

    results = {name: [] for name, _, _ in ALGORITHMS}
    results["LIF SNN"] = []
    lif_firing_log = []  # For the txt output

    for true_angle, servo_pos, fpath in recordings:
        print(f"\n--- True: {true_angle}° | Servo: {servo_pos}° ---")

        recording = np.load(fpath)
        mic_audio = recording[:, 1:5].astype(np.float64)

        # Classical algorithms
        print("    Classical: ", end="", flush=True)
        for name, algo_func, _ in ALGORITHMS:
            try:
                raw_est = algo_func(mic_audio, SAMPLE_RATE)
                cal_offset = calibration_offsets[name]

                if direction_sign == 1:
                    est = (raw_est - cal_offset + 360) % 360
                else:
                    est = (cal_offset - raw_est + 360) % 360

                if est > 180:
                    est = est - 360

                err = abs(est - true_angle)
                if err > 180:
                    err = 360 - err

                results[name].append((true_angle, est, err))
                print(f"{name[:3]}:{est:5.1f}° ", end="")
            except Exception:
                results[name].append((true_angle, None, None))
                print(f"{name[:3]}:ERR ", end="")
        print()

        # LIF SNN
        print("    LIF SNN:  ", end="", flush=True)
        fired_angles, all_fired = evaluate_lif_snn(mic_audio, SAMPLE_RATE, rec_dur)

        if fired_angles:
            # Use the most common firing angle as the estimate
            unique, counts = np.unique(fired_angles, return_counts=True)
            best_angle = unique[np.argmax(counts)]

            # The LIF reports in its own coordinate system — apply same calibration
            # LIF calibration: use the first recording's dominant firing as the offset
            est = best_angle
            err = abs(est - true_angle)
            if err > 180:
                err = 360 - err

            results["LIF SNN"].append((true_angle, est, err))
            print(f"Fired: {best_angle:.0f}° (err={err:.0f}°)", end="")

            for t_sec, a_deg in all_fired:
                lif_firing_log.append((true_angle, servo_pos, t_sec, a_deg))
        else:
            results["LIF SNN"].append((true_angle, None, None))
            print("No neurons fired", end="")

        print()

    # --- Save LIF neuron firings to text file ---
    lif_txt_path = folder / "lif_neuron_firings.txt"
    with open(lif_txt_path, 'w') as f:
        f.write("LIF SNN Neuron Firing Log\n")
        f.write(f"Recording: {folder.name}\n")
        f.write(f"LIF Resolution: {LIF_RESOLUTION_DEG}° per neuron\n")
        f.write("=" * 60 + "\n")
        f.write(f"{'True Angle':>12} {'Servo Pos':>10} {'Fire Time (s)':>14} {'Neuron Angle':>14}\n")
        f.write("-" * 60 + "\n")

        if lif_firing_log:
            for true_ang, servo, t_sec, neuron_ang in lif_firing_log:
                f.write(f"{true_ang:>12d} {servo:>10d} {t_sec:>14.6f} {neuron_ang:>14.1f}\n")
        else:
            f.write("No neurons fired during any recording.\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("Summary by angle:\n")
        f.write(f"{'True Angle':>12} {'Fired Neurons':>40}\n")
        f.write("-" * 60 + "\n")

        # Group by true angle
        from collections import defaultdict
        by_angle = defaultdict(list)
        for true_ang, _, _, neuron_ang in lif_firing_log:
            by_angle[true_ang].append(neuron_ang)

        for true_ang in sorted(by_angle.keys()):
            neurons = by_angle[true_ang]
            unique, counts = np.unique(neurons, return_counts=True)
            summary = ", ".join([f"{a:.0f}°(x{c})" for a, c in zip(unique, counts)])
            f.write(f"{true_ang:>12d} {summary:>40}\n")

        # Angles with no firing
        all_true = set(ta for ta, _, _ in recordings)
        no_fire = sorted(all_true - set(by_angle.keys()))
        if no_fire:
            f.write("\nAngles with no neuron firing:\n")
            for a in no_fire:
                f.write(f"    {a}°\n")

    print(f"\n    LIF firing log saved: {lif_txt_path}")

    # ==========================================================================
    # RESULTS & PLOTTING
    # ==========================================================================

    all_algos = ALGORITHMS + [("LIF SNN", None, LIF_COLOR)]

    print("\n" + "=" * 60)
    print(" Results Summary")
    print("=" * 60)
    for name, _, _ in all_algos:
        valid = [(t, e, err) for t, e, err in results[name] if e is not None]
        if valid:
            errors = [err for _, _, err in valid]
            print(f"    {name:12s}: Mean={np.mean(errors):5.1f}°  "
                  f"Max={np.max(errors):5.1f}°  Std={np.std(errors):4.1f}°")
        else:
            print(f"    {name:12s}: No valid data")

    # --- Plot ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle("DoA Algorithm Comparison (Offline Evaluation)", fontsize=16, fontweight='bold')

    # Plot 1: Tracking accuracy
    ax1 = axes[0, 0]
    for name, _, color in all_algos:
        valid = [(t, e) for t, e, _ in results[name] if e is not None]
        if valid:
            true_vals = [t for t, _ in valid]
            est_vals = [e for _, e in valid]
            ax1.plot(true_vals, est_vals, 'o-', color=color, label=name, markersize=4)
    max_angle = max(t for t, _, _ in recordings)
    ax1.plot([0, max_angle], [0, max_angle], 'k--', linewidth=2, label='Perfect', alpha=0.5)
    ax1.set_xlabel('True Angle (°)')
    ax1.set_ylabel('Estimated Angle (°)')
    ax1.set_title('Tracking Accuracy')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Error comparison
    ax2 = axes[0, 1]
    n_algos = len(all_algos)
    bar_width = 0.8 / n_algos
    for i, (name, _, color) in enumerate(all_algos):
        valid = [(t, err) for t, _, err in results[name] if err is not None]
        if valid:
            x = np.array([t for t, _ in valid]) + i * bar_width
            y = [err for _, err in valid]
            ax2.bar(x, y, width=bar_width, color=color, label=name, alpha=0.7)
    ax2.set_xlabel('True Angle (°)')
    ax2.set_ylabel('Absolute Error (°)')
    ax2.set_title('Error at Each Position')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # Plot 3: Box plot
    ax3 = axes[1, 0]
    error_data = []
    labels = []
    colors = []
    for name, _, color in all_algos:
        valid = [err for _, _, err in results[name] if err is not None]
        if valid:
            error_data.append(valid)
            labels.append(name)
            colors.append(color)
    if error_data:
        bp = ax3.boxplot(error_data, tick_labels=labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    ax3.set_ylabel('Error (°)')
    ax3.set_title('Error Distribution')
    ax3.grid(True, alpha=0.3, axis='y')

    # Plot 4: Mean error bar chart
    ax4 = axes[1, 1]
    means = []
    names_list = []
    cols = []
    for name, _, color in all_algos:
        valid = [err for _, _, err in results[name] if err is not None]
        if valid:
            means.append(np.mean(valid))
            names_list.append(name)
            cols.append(color)
    if means:
        bars = ax4.bar(names_list, means, color=cols, alpha=0.7)
        for bar, val in zip(bars, means):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                     f'{val:.1f}°', ha='center', fontsize=10)
    ax4.set_ylabel('Mean Error (°)')
    ax4.set_title('Algorithm Comparison')
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    print("\n" + "=" * 60)
    plot_name = input("    Enter plot name (or press Enter for 'comparison_results'): ").strip()
    if not plot_name:
        plot_name = "comparison_results"
    if not plot_name.endswith('.png'):
        plot_name += '.png'

    save_path = folder / plot_name
    plt.savefig(save_path, dpi=150)
    print(f"    Plot saved: {save_path}")
    plt.show()

    print("\nDone.")
