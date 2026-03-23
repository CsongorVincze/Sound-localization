import sys
import os
import time
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, resample
from brian2 import *

from kiindulo_kod import create_network, target_angles, deg, c_sound, num_mics, num_neurons

def run_real_audio_sim(duration_sec=5.0):
    print("="*60)
    print(f" Raw Audio LIF Coincidence Simulation ({duration_sec}s) ")
    print("="*60)
    
    SAMPLE_RATE = 16000
    
    # 1. Find ReSpeaker audio device
    respeaker_id = None
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if dev['max_input_channels'] >= 4:
            name = dev['name'].lower()
            if 'respeaker' in name or 'uac1.0' in name or 'seeed' in name:
                respeaker_id = i
                break
                
    if respeaker_id is None:
        print("\nWarning: ReSpeaker not found as a 4-channel audio device. Using default device.")
        respeaker_id = sd.default.device[0]

    device_info = sd.query_devices(respeaker_id, 'input')
    channels = device_info['max_input_channels']
    print(f"\nRecording from '{device_info['name']}' ({channels} channels) at {SAMPLE_RATE}Hz...")
    
    print(f"\n>>> Please clap, whistle, or speak from different directions now! <<<")
    recording = sd.rec(int(duration_sec * SAMPLE_RATE), samplerate=SAMPLE_RATE, 
                       channels=channels, device=respeaker_id, dtype='float32')
    
    # Visual countdown
    for i in range(5):
        sys.stdout.write(f"\rRecording... {(i+1)}s")
        sys.stdout.flush()
        time.sleep(1)
        
    sd.wait()
    print("\nRecording finished. Processing acoustic waves into LIF Spikes...")
    
    # Extract the 4 array mic channels 
    if channels >= 6:
        mics_audio = recording[:, 1:5]
    elif channels >= 4:
        mics_audio = recording[:, 0:4]
    else:
        print(f"Error: Need at least 4 channels to test physical DoA, but device only has {channels}.")
        return

    # 2. Extract Spikes using Peak Detection with High-Fidelity Interpolation
    # 16kHz sample rate introduces a massive quantization jitter of 62.5us which destroys phase tuning.
    # We natively upsample the analog wave to 256kHz to get sub-microsecond physical delay peak accuracy!
    UPSAMPLE_FACTOR = 16
    HIGH_FS = SAMPLE_RATE * UPSAMPLE_FACTOR
    
    print(f"Upsampling digital signals by {UPSAMPLE_FACTOR}x to {HIGH_FS}Hz to resolve exact analog phase diffs...")
    mics_audio_up = resample(mics_audio, len(mics_audio) * UPSAMPLE_FACTOR, axis=0)

    global_max = np.max(np.abs(mics_audio_up))
    threshold = 0.20 * global_max # Increased threshold to 20% to ignore mic rustling/ringing
    
    # Maps Hardware Channels to SNN Matrix indexing
    # Hardware (Clockwise): CH0: 45°, CH1: 315°, CH2: 225°, CH3: 135°
    # kiindulo SNN (CCW):   idx0: 45°, idx1: 135°, idx2: 225°, idx3: 315°
    snn_to_ch_map = {0: 0, 1: 3, 2: 2, 3: 1}
    
    all_raw_spikes = []
    
    for snn_idx in range(4):
        ch_idx = snn_to_ch_map[snn_idx]
        audio_ch = mics_audio_up[:, ch_idx]
        
        # distance mapping 1ms
        min_dist = int(HIGH_FS * 0.001)
        peaks, _ = find_peaks(np.abs(audio_ch), height=threshold, distance=min_dist)
        peak_times_sec = peaks / HIGH_FS
        
        for t in peak_times_sec:
            all_raw_spikes.append((t, snn_idx))
            
    # Sort spikes strictly chronologically for clustering
    all_raw_spikes.sort(key=lambda x: x[0])
    
    # 3. Acoustic "Time Dilation" based on Array Geometry
    # The LIF SNN is hand-tuned for a 10cm x 10cm physical array.
    # The actual ReSpeaker V2 array is 4.65cm square.
    # To use the pure LIF physical coincidence math, we scale the time delays of the micro-events!
    scale_factor = 0.1 / 0.0465
    
    all_indices = []
    all_times = []
    events_t = []
    
    current_cluster = []
    
    def process_cluster(cluster):
        if not cluster: return
        # Calculate the acoustic centroid of the sound event
        centroid = np.mean([s[0] for s in cluster])
        events_t.append(centroid)
        
        # Apply physics scaling solely to the physical relative delays, so absolute time is maintained
        for t, idx in cluster:
            dt = t - centroid
            dt_scaled = dt * scale_factor
            t_sim = centroid + dt_scaled
            all_times.append(t_sim)
            all_indices.append(idx)

    # We cluster spikes that happen within 50ms of each other as a single coherent sound event
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

    print(f"Extracted {len(all_times)} acoustic spikes forming {len(events_t)} distinct sound events.")
    print("Injecting physical delay signals strictly into the LIF coincidence detector...")
    
    # 4. Neural Simulation
    start_scope()
    defaultclock.dt = 1 * us
    
    if all_times:
        # Prevent negative times due to scaling logic edge cases
        min_t = min(all_times)
        if min_t < 0:
            all_times = [t - min_t for t in all_times]
            
        mics = SpikeGeneratorGroup(num_mics, all_indices, all_times * second)
    else:
        print("Empty simulation.")
        mics = SpikeGeneratorGroup(num_mics, [], [] * second)
        
    mott_neurons, synapses, wta_syns = create_network(mics)
    
    spike_mon = SpikeMonitor(mott_neurons)
    state_mon = StateMonitor(mott_neurons, 'v', record=True) 

    print(f"\nRunning Brian2 simulation based purely on sound wave intersections...")
    run(duration_sec * second + 50 * ms, report='text')

    # 5. Visualizer
    print("\nVisualizing Timeline...")
    plt.figure(figsize=(12, 8))

    # ---- Plot 1: Internal Excitation ----
    plt.subplot(2, 1, 1)
    fired_neurons = set(spike_mon.i)
    
    for i in range(num_neurons):
        angle = int(target_angles[i] / deg)
        if i in fired_neurons:
            plt.plot(state_mon.t/second, state_mon.v[i]/mV, color='#2ca02c', linewidth=1.5, zorder=10, label=f'{angle}° (Fired)')
        else:
            plt.plot(state_mon.t/second, state_mon.v[i]/mV, color='grey', alpha=0.3, linewidth=0.8, label=f'{angle}°')

    plt.axhline(y=1000, color='r', linestyle='--', alpha=0.5, label='Threshold (1000 mV)')
    plt.title(f'LIF Neurons Temporal Coincidence Integration')
    plt.ylabel('Excitation Level (mV)')
    
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.grid(True, alpha=0.4)
    plt.xlabel('Time (s)')

    # ---- Plot 2: Spike Output ----
    plt.subplot(2, 1, 2)
    plt.plot(spike_mon.t/second, spike_mon.i, 'X', color='#2ca02c', markersize=12, zorder=5, label='Neuron Fired')
    
    plt.yticks(range(num_neurons), [f'{int(ang/deg)}°' for ang in target_angles])
    plt.xlabel('Time (s)')
    plt.ylabel('Assigned Mott Neuron (Angle)')
    plt.title('Detected Directions from Raw Acoustic Phasing')
    plt.grid(True, axis='y')
    
    for t_sec in events_t:
        plt.axvline(x=t_sec, color='#00d2ff', linestyle=':', alpha=0.8, zorder=1, label='Acoustic Wave Envelopes')
        
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.01, 1), loc='upper left')

    plt.tight_layout()
    # Save the output plot to the current directory
    plt.savefig('real_audio_snn_result.png', dpi=150, bbox_inches='tight')
    print("Plot saved to 'real_audio_snn_result.png'.")
    plt.show()

if __name__ == '__main__':
    run_real_audio_sim()
