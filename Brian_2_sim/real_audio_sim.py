"""
real_audio_sim.py

This script forms the bridge between real-world audio recording (using sounddevice)
and the simulated Brian2 SNN coincidence detection. It processes the raw acoustic
time domain waveforms into precise digital spikes.
"""
import sys
import os
import time
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, resample
from brian2 import *

# We import the algorithm settings directly from the single-responsibility module!
from kiindulo_kod import create_network, target_angles, deg, c_sound, num_mics, num_neurons

def run_real_audio_sim(duration_sec=5.0):
    """
    Captures live audio from a multi-channel ReSpeaker array, translates the phase 
    delays of sound peaks into time-locked spikes, and processes them through the SNN.
    """
    print("="*60)
    print(f" Raw Audio LIF Coincidence Simulation ({duration_sec}s) ")
    print("="*60)
    
    # Standard recording sampling rate
    SAMPLE_RATE = 16000
    
    # ==========================================
    # 1. Hardware Detection & Recording Setup
    # ==========================================
    respeaker_id = None
    devices = sd.query_devices()
    
    # Find the specific ReSpeaker device. We need at least 4 hardware channels to
    # extract relative delays between the 4 discrete microphones.
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
    # Record float32 numpy arrays from the hardware
    recording = sd.rec(int(duration_sec * SAMPLE_RATE), samplerate=SAMPLE_RATE, 
                       channels=channels, device=respeaker_id, dtype='float32')
    
    # Visual countdown while recording takes place
    for i in range(5):
        sys.stdout.write(f"\rRecording... {(i+1)}s")
        sys.stdout.flush()
        time.sleep(1)
        
    sd.wait() # Block until the recording completes
    print("\nRecording finished. Processing acoustic waves into LIF Spikes...")
    
    # Extract the 4 array mic channels by splicing the first subset of tracks.
    if channels >= 6:
        mics_audio = recording[:, 1:5]
    elif channels >= 4:
        mics_audio = recording[:, 0:4]
    else:
        print(f"Error: Need at least 4 channels to test physical DoA, but device only has {channels}.")
        return

    # ==========================================
    # 2. High-Fidelity Spike Extraction Algorithm
    # ==========================================
    # Problem: A 16kHz sample rate takes a reading every 62.5 microseconds. 
    # An acoustic delay between mics might actually be 30 microseconds, which is smaller 
    # than the digital sample rate! This creates massive quantization jitter.
    #
    # Solution: We natively **upsample** the digital wave using mathematical spline/resampling 
    # curve fitting to an effective samplerate of 256kHz (which has ~3.9us resolution).
    UPSAMPLE_FACTOR = 16
    HIGH_FS = SAMPLE_RATE * UPSAMPLE_FACTOR
    
    print(f"Upsampling digital signals by {UPSAMPLE_FACTOR}x to {HIGH_FS}Hz to resolve exact analog phase diffs...")
    mics_audio_up = resample(mics_audio, len(mics_audio) * UPSAMPLE_FACTOR, axis=0)

    # Dynamic Thresholding: Ensure we don't trigger spikes on background hiss
    global_max = np.max(np.abs(mics_audio_up))
    threshold = 0.20 * global_max # Threshold 20% of max loudness
    
    # Cross-Hardware Wiring Diagram:
    # Hardware array might output channels out of order (e.g. mic 0 is top-right, etc.).
    # We map hardware index back to our physics simulation (SNN) CCW coordinate matrix layout.
    snn_to_ch_map = {0: 0, 1: 3, 2: 2, 3: 1}
    
    all_raw_spikes = []
    
    # Execute Peak Detection to define exactly when a sound hit each mic
    for snn_idx in range(4):
        ch_idx = snn_to_ch_map[snn_idx]
        audio_ch = mics_audio_up[:, ch_idx]
        
        # We need a quiet zone (e.g., 1ms cooldown) to avoid double-triggering spikes on ringings
        min_dist = int(HIGH_FS * 0.001)
        peaks, _ = find_peaks(np.abs(audio_ch), height=threshold, distance=min_dist)
        
        peak_times_sec = peaks / HIGH_FS # Convert discrete array index to precise seconds
        
        for t in peak_times_sec:
            all_raw_spikes.append((t, snn_idx))
            
    # Sort spikes strictly chronologically so we can map clustered events safely
    all_raw_spikes.sort(key=lambda x: x[0])
    
    # ==========================================
    # 3. Acoustic "Time Dilation" Matrix
    # ==========================================
    # The LIF SNN is hand-tuned in kiindulo_kod for a 10cm physical mic array base geometry.
    # However, the physical external ReSpeaker array is technically 4.65cm. 
    # By artificially blowing up the relative time differences between the mics, 
    # we make the 4.65cm array "look" identical to a 10cm array to our mathematics!
    scale_factor = 0.1 / 0.0465
    
    all_indices = []
    all_times = []
    events_t = []
    
    current_cluster = []
    
    def process_cluster(cluster):
        if not cluster: return
        # A cluster of spikes represents a single physical sound event.
        # Calculate the median acoustic centroid time to maintain accurate time axes
        centroid = np.mean([s[0] for s in cluster])
        events_t.append(centroid)
        
        # Scale only the *differences* in phase away from the centroid
        for t, idx in cluster:
            dt = t - centroid
            dt_scaled = dt * scale_factor
            t_sim = centroid + dt_scaled
            
            all_times.append(t_sim)
            all_indices.append(idx)

    # Sound wave clusters usually last ~50 milliseconds. We logically isolate them.
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
    
    # ==========================================
    # 4. Neural Simulation Execution
    # ==========================================
    start_scope()
    defaultclock.dt = 1 * us
    
    # We must format everything perfectly to avoid Brian2 crash states
    if all_times:
        min_t = min(all_times)
        if min_t < 0:
            all_times = [t - min_t for t in all_times]
            
        mics = SpikeGeneratorGroup(num_mics, all_indices, all_times * second)
    else:
        print("Empty simulation. No loud peaks were found.")
        mics = SpikeGeneratorGroup(num_mics, [], [] * second)
        
    # Plug the spike definitions into our decoupled LIF structure
    mott_neurons, synapses, wta_syns = create_network(mics)
    
    spike_mon = SpikeMonitor(mott_neurons)
    state_mon = StateMonitor(mott_neurons, 'v', record=True) 

    print(f"\nRunning Brian2 simulation based purely on sound wave intersections...")
    # Plus buffer time to finish logic
    run(duration_sec * second + 50 * ms, report='text')

    # ==========================================
    # 5. Visualizer Result Generation
    # ==========================================
    print("\nVisualizing Timeline...")
    plt.figure(figsize=(12, 8))

    # ---- Plot 1: Internal SNN Sub-Threshold Excitation ----
    plt.subplot(2, 1, 1)
    fired_neurons = set(spike_mon.i)
    
    for i in range(num_neurons):
        angle = int(target_angles[i] / deg)
        if i in fired_neurons:
            # Emphasize neurons that fired (achieved coincidence)
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

    # ---- Plot 2: Spike Output Vector Over Time ----
    plt.subplot(2, 1, 2)
    plt.plot(spike_mon.t/second, spike_mon.i, 'X', color='#2ca02c', markersize=12, zorder=5, label='Neuron Fired')
    
    plt.yticks(range(num_neurons), [f'{int(ang/deg)}°' for ang in target_angles])
    plt.xlabel('Time (s)')
    plt.ylabel('Assigned Mott Neuron (Angle)')
    plt.title('Detected Directions from Raw Acoustic Phasing')
    plt.grid(True, axis='y')
    
    # Overlay light blue background lines representing acoustic events
    for t_sec in events_t:
        plt.axvline(x=t_sec, color='#00d2ff', linestyle=':', alpha=0.8, zorder=1, label='Acoustic Wave Envelopes')
        
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.01, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig('real_audio_snn_result.png', dpi=150, bbox_inches='tight')
    print("Plot saved to 'real_audio_snn_result.png'.")
    plt.show()

if __name__ == '__main__':
    run_real_audio_sim()
