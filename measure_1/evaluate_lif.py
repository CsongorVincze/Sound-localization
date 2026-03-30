"""
evaluate_lif.py — LIF SNN Offline Evaluation with Parameter Sweep

Loads pre-recorded audio from the recordings folder and evaluates them
using the Brian2 LIF coincidence detection network from kiindulo_kod.py.

Runs multiple parameter configurations (tau_leaky, v_thresh) and compares
which neuron fired vs. the actual DoA angle for each recording.

Outputs per parameter set:
  - lif_firings_<params>.txt — which neuron fired at which angle
  - Summary statistics

Final output:
  - Comparison plot across all parameter sets
  - Best parameter recommendation

Usage:
    python evaluate_lif.py                                         # both folders
    python evaluate_lif.py recordings/Goldberg_3sec/Goldberg       # specific
"""
import sys
import os
import csv
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# Add paths
MEASURE_DIR = Path(__file__).parent
PROJECT_DIR = MEASURE_DIR.parent
sys.path.insert(0, str(MEASURE_DIR))
sys.path.insert(0, str(PROJECT_DIR / "Brian_2_sim"))

# =============================================================================
# CONFIGURATION
# =============================================================================
SAMPLE_RATE = 16000
LIF_RESOLUTION_DEG = 15

# Parameter sets to evaluate:
# (name, tau_leaky_us, v_thresh_V)
# Based on optimization results and variations to explore
PARAM_SETS = [
    ("optimized",      58,   0.80),   # From sweep_angles.py optimal
    ("respeaker",      55,   1.00),   # From real_audio_sim.py (ReSpeaker-tuned)
    ("tight_leak",     40,   0.80),   # Very fast decay — needs near-perfect coincidence
    ("loose_leak",    100,   0.80),   # Slower decay — more forgiving timing
    ("low_thresh",     58,   0.50),   # Lower threshold — fires more easily
    ("high_thresh",    58,   1.20),   # Higher threshold — only strong coincidences
]

# =============================================================================
# LIF EVALUATION FUNCTION
# =============================================================================

def evaluate_lif_on_recording(mic_audio, sample_rate, duration_sec,
                               tau_leaky_us, v_thresh_V):
    """
    Run the Brian2 LIF coincidence detection on a single 4-channel audio recording.

    Returns:
        fired_angles: list of neuron angles (degrees) that fired
        all_fired: list of (time_sec, neuron_angle_deg)
        num_spikes_extracted: how many input spikes were found
    """
    from scipy.signal import find_peaks, resample
    from brian2 import (start_scope, defaultclock, us, ms, second, meter,
                        volt, mV, SpikeGeneratorGroup, SpikeMonitor, run as brian_run,
                        array, cos, sin, max, min)
    from kiindulo_kod import create_network, get_respeaker_array_geometry, get_target_angles

    start_scope()
    defaultclock.dt = 1 * us

    mic_x, mic_y, num_mics = get_respeaker_array_geometry()
    target_angles, deg = get_target_angles(LIF_RESOLUTION_DEG)
    num_neurons = len(target_angles)
    c_sound = 343 * meter / second

    # --- Spike extraction (same pipeline as real_audio_sim.py) ---
    UPSAMPLE_FACTOR = 16
    HIGH_FS = sample_rate * UPSAMPLE_FACTOR

    mics_audio_up = resample(mic_audio, len(mic_audio) * UPSAMPLE_FACTOR, axis=0)

    global_max = np.max(np.abs(mics_audio_up))
    if global_max < 1e-10:
        return [], [], 0
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

    # Clustering (50ms window)
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

    num_input_spikes = len(all_times)
    if num_input_spikes == 0:
        return [], [], 0

    min_t = np.min(all_times)
    if min_t < 0:
        all_times = [t - min_t for t in all_times]

    mics_sg = SpikeGeneratorGroup(num_mics, all_indices,
                                   np.array(all_times) * second)

    mott_neurons, synapses, wta_syns = create_network(
        mics=mics_sg,
        mic_x=mic_x,
        mic_y=mic_y,
        target_angles=target_angles,
        tau_leaky=tau_leaky_us * us,
        v_thresh=v_thresh_V * volt,
        c_sound=c_sound
    )

    spike_mon = SpikeMonitor(mott_neurons)
    brian_run(duration_sec * second + 50 * ms, report='text')

    fired_angles = []
    all_fired = []
    for spike_idx, spike_t in zip(spike_mon.i, spike_mon.t):
        angle_deg = float(target_angles[int(spike_idx)] / deg)
        fired_angles.append(angle_deg)
        all_fired.append((float(spike_t / second), angle_deg))

    return fired_angles, all_fired, num_input_spikes


# =============================================================================
# PROCESS ONE SOUND FOLDER
# =============================================================================

def process_folder(folder_path, param_sets):
    """Evaluate all recordings in a folder with multiple parameter sets."""
    folder = Path(folder_path)
    sound_name = folder.name

    npy_files = sorted(folder.glob("angle_*.npy"))
    if not npy_files:
        print(f"    No angle_*.npy files in {folder}")
        return None

    # Parse recordings
    recordings = []
    for f in npy_files:
        parts = f.stem.split("_")
        true_angle = int(parts[1])
        servo_pos = int(parts[3])
        recordings.append((true_angle, servo_pos, f))

    print(f"\n    Found {len(recordings)} recordings (0°–{recordings[-1][0]}°)")

    # Load metadata
    meta_path = folder / "metadata.npy"
    if meta_path.exists():
        meta = np.load(meta_path, allow_pickle=True).item()
        rec_dur = meta.get('recording_duration', 3.0)
    else:
        rec_dur = 3.0

    # Results: {param_name: [(true_angle, best_fired, error, n_fired, n_input_spikes), ...]}
    all_results = {}

    for pi, (param_name, tau_us, v_thresh) in enumerate(param_sets):
        print(f"\n{'='*60}")
        print(f" [{pi+1}/{len(param_sets)}] {sound_name} | τ={tau_us}us, V_thresh={v_thresh}V  ({param_name})")
        print(f"{'='*60}")

        results = []
        firing_log = []
        param_start = time.time()

        for ri, (true_angle, servo_pos, fpath) in enumerate(recordings):
            rec_start = time.time()
            print(f"  [{ri+1}/{len(recordings)}] {true_angle:3d}°: ", end="", flush=True)

            recording = np.load(fpath)
            mic_audio = recording[:, 1:5].astype(np.float64)

            fired_angles, all_fired, n_input = evaluate_lif_on_recording(
                mic_audio, SAMPLE_RATE, rec_dur, tau_us, v_thresh
            )

            elapsed = time.time() - rec_start

            if fired_angles:
                unique, counts = np.unique(fired_angles, return_counts=True)
                best_angle = unique[np.argmax(counts)]
                total_fires = len(fired_angles)

                err = abs(best_angle - true_angle)
                if err > 180:
                    err = 360 - err

                results.append((true_angle, best_angle, err, total_fires, n_input))
                summary = ", ".join([f"{a:.0f}°(x{c})" for a, c in zip(unique, counts)])
                print(f"→ {best_angle:5.0f}° (err={err:4.0f}°) | fired: {summary} | spikes_in={n_input} | {elapsed:.1f}s")

                for t_sec, a_deg in all_fired:
                    firing_log.append((true_angle, servo_pos, t_sec, a_deg))
            else:
                results.append((true_angle, None, None, 0, n_input))
                print(f"→ NO FIRE | spikes_in={n_input} | {elapsed:.1f}s")

        param_elapsed = time.time() - param_start
        valid_count = sum(1 for _, _, err, _, _ in results if err is not None)
        print(f"\n  Param set '{param_name}' done in {param_elapsed:.0f}s "
              f"({valid_count}/{len(results)} fired)")

        # --- Calibrate LIF: use first param set's 0° result as offset ---
        if pi == 0:
            # Find what the LIF reports at true_angle=0
            lif_cal_offset = None
            for ta, bp, _, _, _ in results:
                if ta == 0 and bp is not None:
                    lif_cal_offset = bp
                    break
            if lif_cal_offset is None:
                lif_cal_offset = 180.0
                print(f"  WARNING: No LIF data at 0°, using offset={lif_cal_offset}")
            else:
                print(f"  LIF calibration: raw={lif_cal_offset:.0f}° at true 0°")
            print(f"  Correction: corrected = ({lif_cal_offset:.0f} - raw) % 360")

        corrected = []
        for ta, bp, _, nf, ni in results:
            if bp is not None:
                bp_corr = (lif_cal_offset - bp) % 360
                if bp_corr > 180:
                    bp_corr = bp_corr - 360
                err = abs(bp_corr - ta)
                if err > 180:
                    err = 360 - err
                corrected.append((ta, bp_corr, err, nf, ni))
            else:
                corrected.append((ta, None, None, nf, ni))
        results = corrected

        all_results[param_name] = results

        # --- Save firing log to txt ---
        txt_path = folder / f"lif_firings_{param_name}.txt"
        with open(txt_path, 'w') as f:
            f.write(f"LIF SNN Neuron Firing Log — {sound_name}\n")
            f.write(f"Parameters: tau_leaky={tau_us}us, v_thresh={v_thresh}V\n")
            f.write(f"LIF Resolution: {LIF_RESOLUTION_DEG}° per neuron\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 70 + "\n")
            f.write(f"{'True Angle':>12} {'Servo Pos':>10} {'Fire Time (s)':>14} {'Neuron Angle':>14}\n")
            f.write("-" * 70 + "\n")

            if firing_log:
                for ta, sp, ts, na in firing_log:
                    f.write(f"{ta:>12d} {sp:>10d} {ts:>14.6f} {na:>14.1f}\n")
            else:
                f.write("No neurons fired.\n")

            f.write("\n" + "=" * 70 + "\n")
            f.write("Summary:\n")
            f.write(f"{'True Angle':>12} {'Best Pred':>10} {'Error':>8} {'#Fires':>8} {'#Input':>8}\n")
            f.write("-" * 70 + "\n")
            for ta, bp, err, nf, ni in results:
                bp_str = f"{bp:.0f}°" if bp is not None else "NONE"
                err_str = f"{err:.0f}°" if err is not None else "-"
                f.write(f"{ta:>12d} {bp_str:>10} {err_str:>8} {nf:>8d} {ni:>8d}\n")

            valid = [(ta, err) for ta, _, err, _, _ in results if err is not None]
            if valid:
                errors = [e for _, e in valid]
                f.write(f"\nMean error: {np.mean(errors):.1f}°\n")
                f.write(f"Max error:  {np.max(errors):.1f}°\n")
                f.write(f"Std error:  {np.std(errors):.1f}°\n")
                f.write(f"Success rate: {len(valid)}/{len(results)} "
                        f"({100*len(valid)/len(results):.0f}%)\n")

        print(f"  → Saved: {txt_path.name}")

        # --- Save raw results as .npy (crash-safe — saved after EACH param set) ---
        npy_path = folder / f"lif_results_{param_name}.npy"
        np.save(npy_path, {
            'param_name': param_name,
            'tau_leaky_us': tau_us,
            'v_thresh_V': v_thresh,
            'results': results,        # [(true_angle, best_fired, error, n_fired, n_input), ...]
            'firing_log': firing_log,   # [(true_angle, servo_pos, time_sec, neuron_angle), ...]
            'sound_name': sound_name,
            'timestamp': datetime.now().isoformat(),
        })
        print(f"  → Saved: {npy_path.name}")

    # --- Save combined CSV summary across all parameter sets ---
    csv_path = folder / f"lif_summary_{sound_name}.csv"
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['true_angle', 'param_name', 'tau_us', 'v_thresh',
                         'predicted_angle', 'error', 'n_fires', 'n_input_spikes'])
        for param_name, tau_us, v_thresh in param_sets:
            if param_name not in all_results:
                continue
            for ta, bp, err, nf, ni in all_results[param_name]:
                writer.writerow([ta, param_name, tau_us, v_thresh,
                                 bp if bp is not None else '',
                                 err if err is not None else '',
                                 nf, ni])
    print(f"  → Saved: {csv_path.name}")

    # --- Save grand summary txt ---
    grand_path = folder / f"lif_grand_summary_{sound_name}.txt"
    with open(grand_path, 'w') as f:
        f.write(f"LIF SNN Grand Summary — {sound_name}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Recordings: {len(recordings)} angles (0°–{recordings[-1][0]}°)\n")
        f.write(f"LIF Resolution: {LIF_RESOLUTION_DEG}° per neuron\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"{'Parameter Set':<16} {'tau(us)':<10} {'Vt(V)':<8} "
                f"{'Mean Err':<10} {'Max Err':<10} {'Std Err':<10} "
                f"{'Success':<12} {'#Fired':<10}\n")
        f.write("-" * 80 + "\n")

        best_mean = float('inf')
        best_name = ""

        for param_name, tau_us, v_thresh in param_sets:
            if param_name not in all_results:
                continue
            res = all_results[param_name]
            valid = [(ta, err) for ta, _, err, _, _ in res if err is not None]
            total_fires = sum(nf for _, _, _, nf, _ in res)

            if valid:
                errors = [e for _, e in valid]
                me = np.mean(errors)
                mx = np.max(errors)
                sd = np.std(errors)
                sr = f"{len(valid)}/{len(res)} ({100*len(valid)/len(res):.0f}%)"
                if me < best_mean:
                    best_mean = me
                    best_name = param_name
            else:
                me, mx, sd = 0, 0, 0
                sr = f"0/{len(res)} (0%)"

            f.write(f"{param_name:<16} {tau_us:<10} {v_thresh:<8.2f} "
                    f"{me:<10.1f} {mx:<10.1f} {sd:<10.1f} "
                    f"{sr:<12} {total_fires:<10}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write(f"Best parameter set: {best_name} (mean error = {best_mean:.1f}°)\n")

        # Per-angle breakdown table
        f.write("\n" + "=" * 80 + "\n")
        f.write("Per-angle breakdown (predicted angle for each param set):\n")
        f.write("-" * 80 + "\n")
        header = f"{'True°':<8}" + "".join([f"{pn:<14}" for pn, _, _ in param_sets])
        f.write(header + "\n")
        f.write("-" * 80 + "\n")
        for idx, (ta, _, _) in enumerate(recordings):
            line = f"{ta:<8}"
            for pn, _, _ in param_sets:
                if pn in all_results and idx < len(all_results[pn]):
                    _, bp, err, _, _ = all_results[pn][idx]
                    if bp is not None:
                        line += f"{bp:>5.0f}°(e={err:.0f}°)  "
                    else:
                        line += f"{'NO FIRE':<14}"
                else:
                    line += f"{'-':<14}"
            f.write(line + "\n")

    print(f"  → Saved: {grand_path.name}")

    return all_results, recordings, sound_name


# =============================================================================
# PLOTTING
# =============================================================================

def plot_comparison(all_results, recordings, sound_name, param_sets, save_dir):
    """Create comparison plot across parameter sets."""
    true_angles = [ta for ta, _, _ in recordings]
    max_angle = max(true_angles)

    n_params = len(param_sets)
    colors = plt.cm.tab10(np.linspace(0, 1, n_params))

    fig, axes = plt.subplots(2, 2, figsize=(18, 13))
    fig.suptitle(f"LIF SNN Parameter Comparison — {sound_name}",
                 fontsize=16, fontweight='bold')

    # 1. Tracking accuracy
    ax1 = axes[0, 0]
    ax1.plot([0, max_angle], [0, max_angle], 'k--', lw=2, alpha=0.4, label='Perfect')
    for i, (pname, tau, vt) in enumerate(param_sets):
        if pname not in all_results:
            continue
        valid = [(ta, bp) for ta, bp, _, _, _ in all_results[pname] if bp is not None]
        if valid:
            t_vals = [t for t, _ in valid]
            p_vals = [p for _, p in valid]
            ax1.plot(t_vals, p_vals, 'o-', color=colors[i], markersize=4,
                     label=f"{pname} (τ={tau}, Vt={vt})")
    ax1.set_xlabel('True Angle (°)')
    ax1.set_ylabel('LIF Predicted Angle (°)')
    ax1.set_title('Tracking Accuracy')
    ax1.legend(fontsize=8, loc='upper left')
    ax1.grid(True, alpha=0.3)

    # 2. Error per angle
    ax2 = axes[0, 1]
    bar_w = 0.8 / n_params
    for i, (pname, tau, vt) in enumerate(param_sets):
        if pname not in all_results:
            continue
        valid = [(ta, err) for ta, _, err, _, _ in all_results[pname] if err is not None]
        if valid:
            x = np.array([t for t, _ in valid]) + i * bar_w
            y = [e for _, e in valid]
            ax2.bar(x, y, width=bar_w, color=colors[i], alpha=0.7,
                    label=f"{pname}")
    ax2.set_xlabel('True Angle (°)')
    ax2.set_ylabel('Absolute Error (°)')
    ax2.set_title('Error at Each Position')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. Box plot
    ax3 = axes[1, 0]
    error_data = []
    box_labels = []
    box_colors = []
    for i, (pname, tau, vt) in enumerate(param_sets):
        if pname not in all_results:
            continue
        valid = [err for _, _, err, _, _ in all_results[pname] if err is not None]
        if valid:
            error_data.append(valid)
            box_labels.append(f"{pname}\nτ={tau},Vt={vt}")
            box_colors.append(colors[i])
    if error_data:
        bp = ax3.boxplot(error_data, tick_labels=box_labels, patch_artist=True)
        for patch, c in zip(bp['boxes'], box_colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
    ax3.set_ylabel('Error (°)')
    ax3.set_title('Error Distribution')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.tick_params(axis='x', labelsize=7)

    # 4. Summary bar chart (mean error + success rate)
    ax4 = axes[1, 1]
    mean_errors = []
    success_rates = []
    bar_labels = []
    bar_colors = []
    for i, (pname, tau, vt) in enumerate(param_sets):
        if pname not in all_results:
            continue
        res = all_results[pname]
        valid = [err for _, _, err, _, _ in res if err is not None]
        if valid:
            mean_errors.append(np.mean(valid))
            success_rates.append(100 * len(valid) / len(res))
        else:
            mean_errors.append(90)
            success_rates.append(0)
        bar_labels.append(f"{pname}\nτ={tau}")
        bar_colors.append(colors[i])

    x_pos = np.arange(len(bar_labels))
    bars = ax4.bar(x_pos, mean_errors, color=bar_colors, alpha=0.7)
    for j, (bar, me, sr) in enumerate(zip(bars, mean_errors, success_rates)):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f'{me:.1f}°\n({sr:.0f}%)', ha='center', fontsize=8)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(bar_labels, fontsize=7)
    ax4.set_ylabel('Mean Error (°)')
    ax4.set_title('Mean Error & Success Rate')
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    plot_path = save_dir / f"lif_comparison_{sound_name}.png"
    plt.savefig(plot_path, dpi=150)
    print(f"\n    Plot saved: {plot_path}")

    # Also save plot data as .npy so it can be replotted later
    plot_data_path = save_dir / f"lif_plot_data_{sound_name}.npy"
    np.save(plot_data_path, {
        'all_results': all_results,
        'recordings': [(ta, sp) for ta, sp, _ in recordings],
        'sound_name': sound_name,
        'param_sets': param_sets,
        'timestamp': datetime.now().isoformat(),
    })
    print(f"    Plot data saved: {plot_data_path}")

    plt.show()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')

    from brian2 import prefs
    prefs.codegen.target = 'numpy'

    print("=" * 60)
    print(" LIF SNN Offline Evaluation — Parameter Comparison")
    print("=" * 60)

    # Determine which folders to process
    recordings_root = MEASURE_DIR / "recordings"

    if len(sys.argv) > 1:
        # Explicit folder(s) given
        folders_to_process = [Path(a) for a in sys.argv[1:]]
    else:
        # Auto-detect: look for subfolders with angle_*.npy files
        folders_to_process = []
        for session in sorted(recordings_root.iterdir()):
            if session.is_dir():
                # Check if this folder directly has recordings
                if list(session.glob("angle_*.npy")):
                    folders_to_process.append(session)
                else:
                    # Check subfolders (multi-sound sessions)
                    for sub in sorted(session.iterdir()):
                        if sub.is_dir() and list(sub.glob("angle_*.npy")):
                            folders_to_process.append(sub)

    if not folders_to_process:
        print("ERROR: No recording folders found!")
        print(f"  Searched in: {recordings_root}")
        sys.exit(1)

    print(f"\nFolders to evaluate:")
    for f in folders_to_process:
        print(f"    • {f}")

    print(f"\nParameter sets to test:")
    for name, tau, vt in PARAM_SETS:
        print(f"    • {name}: τ_leak={tau}us, V_thresh={vt}V")

    # Process each folder
    all_saved_files = []

    for folder in folders_to_process:
        result = process_folder(folder, PARAM_SETS)
        if result is not None:
            all_results, recordings, sound_name = result
            plot_comparison(all_results, recordings, sound_name,
                           PARAM_SETS, folder)

            # Collect saved files for final summary
            for f in folder.glob("lif_*"):
                all_saved_files.append(f)

    print("\n" + "=" * 60)
    print(" All evaluations complete!")
    print("=" * 60)
    print(f"\n  Saved {len(all_saved_files)} output files:")
    for f in sorted(all_saved_files):
        size_kb = f.stat().st_size / 1024
        print(f"    {f.relative_to(MEASURE_DIR)}  ({size_kb:.1f} KB)")
    print()
