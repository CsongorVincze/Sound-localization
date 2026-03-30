"""
evaluate_single.py — Run LIF coincidence detection on a single pre-recorded audio file.

Loads one .npy recording, extracts spikes, runs the Brian2 SNN, and plots:
  1. Sub-threshold membrane voltage traces (all neurons)
  2. Neuron spike raster (which neuron fired at what time)

Usage:
    python evaluate_single.py                           # uses default file
    python evaluate_single.py path/to/angle_045.npy     # specify recording
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import find_peaks, resample

# Ensure Brian_2_sim is importable
sys.path.insert(0, str(Path(__file__).parent))

from brian2 import *
prefs.codegen.target = 'numpy'

from kiindulo_kod import create_network, get_respeaker_array_geometry, get_target_angles


def run_lif_on_file(npy_path,
                    sample_rate=16000,
                    duration_sec=3.0,
                    tau_leaky_us=55,
                    v_thresh_V=1.0,
                    upsample_factor=16,
                    spike_thresh_frac=0.20):
    """
    Full LIF evaluation pipeline on one .npy recording.

    Args:
        npy_path:          Path to a multi-channel .npy audio recording.
        sample_rate:       Original recording sample rate (Hz).
        duration_sec:      Duration of the recording (seconds).
        tau_leaky_us:      LIF leak time constant (microseconds).
        v_thresh_V:        LIF spike threshold (volts).
        upsample_factor:   How much to upsample for phase precision.
        spike_thresh_frac: Fraction of global max used as peak threshold.

    Returns:
        spike_mon:      Brian2 SpikeMonitor (neuron firings)
        state_mon:      Brian2 StateMonitor (voltage traces)
        target_angles:  Array of neuron angles (radians)
        deg:            Degrees conversion factor
        events_t:       Centroids of acoustic event clusters (seconds)
        n_input_spikes: Number of input spikes extracted
    """
    print(f"\nLoading: {npy_path}")
    recording = np.load(npy_path)
    print(f"  Shape: {recording.shape}  (samples x channels)")

    # Extract the 4 microphone channels (ReSpeaker 6-ch layout: ch 1–4)
    if recording.shape[1] >= 6:
        mic_audio = recording[:, 1:5].astype(np.float64)
    elif recording.shape[1] >= 4:
        mic_audio = recording[:, 0:4].astype(np.float64)
    else:
        raise ValueError(f"Need at least 4 channels, got {recording.shape[1]}")

    # ---- Physical array geometry ----
    mic_x, mic_y, num_mics = get_respeaker_array_geometry()
    target_angles, deg = get_target_angles(15)
    num_neurons = len(target_angles)
    c_sound = 343 * meter / second

    # ---- Upsample for phase precision ----
    HIGH_FS = sample_rate * upsample_factor
    print(f"  Upsampling {sample_rate}Hz → {HIGH_FS}Hz ({upsample_factor}x)")
    mics_up = resample(mic_audio, len(mic_audio) * upsample_factor, axis=0)

    # ---- Spike extraction (same as real_audio_sim.py) ----
    global_max = np.max(np.abs(mics_up))
    if global_max < 1e-10:
        raise ValueError("Recording is silent (global max ≈ 0).")

    threshold = spike_thresh_frac * global_max
    print(f"  Peak threshold: {threshold:.6f} ({spike_thresh_frac*100:.0f}% of max {global_max:.6f})")

    snn_to_ch_map = {0: 0, 1: 3, 2: 2, 3: 1}

    all_raw_spikes = []
    for snn_idx in range(4):
        ch_idx = snn_to_ch_map[snn_idx]
        audio_ch = mics_up[:, ch_idx]
        min_dist = int(HIGH_FS * 0.001)  # 1ms refractory
        peaks, _ = find_peaks(np.abs(audio_ch), height=threshold, distance=min_dist)
        for t in peaks / HIGH_FS:
            all_raw_spikes.append((t, snn_idx))

    all_raw_spikes.sort(key=lambda x: x[0])

    # ---- Clustering (50ms window) ----
    all_indices = []
    all_times = []
    events_t = []
    current_cluster = []

    def process_cluster(cluster):
        if not cluster:
            return
        centroid = np.mean([s[0] for s in cluster])
        events_t.append(centroid)
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

    n_input_spikes = len(all_times)
    print(f"  Extracted {n_input_spikes} spikes in {len(events_t)} acoustic events")

    if n_input_spikes == 0:
        raise ValueError("No spikes extracted — recording may be too quiet.")

    # Ensure non-negative times
    min_t = min(all_times)
    if min_t < 0:
        all_times = [t - min_t for t in all_times]

    # ---- Brian2 simulation ----
    start_scope()
    defaultclock.dt = 1 * us

    mics_sg = SpikeGeneratorGroup(num_mics, all_indices, np.array(all_times) * second)

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
    state_mon = StateMonitor(mott_neurons, 'v', record=True)

    print(f"\n  Running Brian2 SNN (duration={duration_sec}s + 50ms buffer)...")
    run(duration_sec * second + 50 * ms, report='text')

    return spike_mon, state_mon, target_angles, deg, events_t, n_input_spikes


def plot_results(spike_mon, state_mon, target_angles, deg, events_t,
                 n_input_spikes, title_extra="", save_path=None):
    """
    Two-panel plot:
      Top:    membrane voltage traces (green = fired, grey = silent)
      Bottom: spike raster (which neuron fired when)
    """
    num_neurons = len(target_angles)
    fired_neurons = set(spike_mon.i)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), 
                                    gridspec_kw={'height_ratios': [1.2, 1]})
    fig.suptitle(f'LIF Coincidence Detection — Single Recording{title_extra}',
                 fontsize=14, fontweight='bold')

    # ---- Top: voltage traces ----
    for i in range(num_neurons):
        angle = int(target_angles[i] / deg)
        if i in fired_neurons:
            ax1.plot(state_mon.t / second, state_mon.v[i] / mV,
                     color='#2ca02c', linewidth=1.5, zorder=10,
                     label=f'{angle}° (Fired)')
        else:
            ax1.plot(state_mon.t / second, state_mon.v[i] / mV,
                     color='grey', alpha=0.3, linewidth=0.8)

    ax1.axhline(y=1000, color='r', linestyle='--', alpha=0.5, label='Threshold (1000 mV)')
    ax1.set_ylabel('Membrane Voltage (mV)')
    ax1.set_title('Sub-Threshold Excitation of All Neurons')

    # De-duplicate legend
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys(),
               bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.4)

    # ---- Bottom: spike raster ----
    if len(spike_mon.t) > 0:
        ax2.plot(spike_mon.t / second, spike_mon.i, 'X',
                 color='#2ca02c', markersize=12, zorder=5, label='Neuron Fired')

    ax2.set_yticks(range(num_neurons))
    ax2.set_yticklabels([f'{int(ang / deg)}°' for ang in target_angles], fontsize=7)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Neuron (Target Angle)')
    ax2.set_title('Detected Directions from Acoustic Phase Differences')
    ax2.grid(True, axis='y', alpha=0.4)

    # Overlay acoustic event centroids
    for t_sec in events_t:
        ax2.axvline(x=t_sec, color='#00d2ff', linestyle=':', alpha=0.6, zorder=1)
    # Manual legend entry for event lines
    if events_t:
        ax2.axvline(x=-1, color='#00d2ff', linestyle=':', alpha=0.8, label='Acoustic Events')

    handles2, labels2 = ax2.get_legend_handles_labels()
    by_label2 = dict(zip(labels2, handles2))
    if by_label2:
        ax2.legend(by_label2.values(), by_label2.keys(),
                   bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")

    plt.show()

    # ---- Console summary ----
    print(f"\n{'='*50}")
    print(f" Results Summary")
    print(f"{'='*50}")
    print(f"  Input spikes:    {n_input_spikes}")
    print(f"  Acoustic events: {len(events_t)}")
    print(f"  Neurons fired:   {len(fired_neurons)}")

    if fired_neurons:
        fired_angles = [int(target_angles[int(i)] / deg) for i in spike_mon.i]
        unique, counts = np.unique(fired_angles, return_counts=True)
        print(f"  Firing breakdown:")
        for a, c in zip(unique, counts):
            print(f"    {a:4d}° — {c} spike(s)")

        best = unique[np.argmax(counts)]
        print(f"\n  → Strongest direction: {best}°")
    else:
        print("  No neurons fired. Try lowering v_thresh or raising tau_leaky.")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')

    # Default recording path (change as needed)
    DEFAULT_FILE = str(Path(__file__).parent.parent /
                       "measure_1" / "recordings" / "Goldberg_3sec" /
                       "Goldberg" / "angle_045_servo_090.npy")

    if len(sys.argv) > 1:
        npy_file = sys.argv[1]
    else:
        npy_file = DEFAULT_FILE

    if not os.path.exists(npy_file):
        print(f"ERROR: File not found: {npy_file}")
        sys.exit(1)

    # Parse angle from filename for the title
    fname = Path(npy_file).stem
    parts = fname.split('_')
    try:
        true_angle = int(parts[1])
        title_extra = f"  |  True angle: {true_angle}°"
    except (IndexError, ValueError):
        true_angle = None
        title_extra = ""

    save_name = Path(npy_file).parent / f"lif_single_{fname}.png"

    spike_mon, state_mon, target_angles, deg, events_t, n_input = run_lif_on_file(
        npy_file,
        tau_leaky_us=55,
        v_thresh_V=1.0,
    )

    plot_results(spike_mon, state_mon, target_angles, deg, events_t,
                 n_input, title_extra=title_extra, save_path=str(save_name))
