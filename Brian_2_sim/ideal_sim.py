"""
Simulates an ideal acoustic point source for testing the Leaky Integrate-and-Fire 
(LIF) network's coincidence detection mapping. This isolates the testing logic 
from the core algorithm.
"""
from brian2 import *
import numpy as np
import matplotlib.pyplot as plt

# Import the core network and necessary helper functions
from kiindulo_kod import create_network, get_default_array_geometry, get_target_angles

def run_simulation(true_angle_deg=67, plot_results=True):
    """
    Simulates a perfect acoustic event arriving from a specific angle, feeds it into 
    the LIF network, and plots the internal voltage traces of the neurons.
    """
    start_scope() # Reset Brian2 simulator memory for a fresh run
    defaultclock.dt = 1 * us # Time step must be very small to resolve sub-millisecond physical delays

    # ==========================================
    # 0. Load Hardware Constraints (Modular params)
    # ==========================================
    mic_x, mic_y, num_mics = get_default_array_geometry()
    target_angles, deg = get_target_angles(15)
    num_neurons = len(target_angles)
    c_sound = 343 * meter / second
    
    # ==========================================
    # 1. Simulate the Ideal Acoustic Event
    # ==========================================
    true_angle = true_angle_deg * deg # The actual incident angle of the sound

    # Calculate theoretical arrival times at each microphone coordinate,
    # based on the dot product of the wave vector and mic position.
    t_arrival = -(mic_x * cos(true_angle) + mic_y * sin(true_angle)) / c_sound
    t_arrival -= min(t_arrival) # Shift baseline so the first mic gets hit exactly at t=0
    t_arrival += 5 * ms # Add a buffer time of 5ms to allow the simulation to settle

    # Encode these arrival times as Brian2 Spikes.
    indices = array([0, 1, 2, 3])
    mics = SpikeGeneratorGroup(num_mics, indices, t_arrival)

    # ==========================================
    # 2. Setup the Spiking Neural Network (SNN)
    # ==========================================
    # Instantiate our purely algorithmic LIF model mapping the variables we just declared.
    # Note: we can now dynamically pass in tau_leaky, speeds, or custom weights if we wanted!
    mott_neurons, synapses, wta_syns = create_network(mics=mics, 
                                                      mic_x=mic_x, 
                                                      mic_y=mic_y, 
                                                      target_angles=target_angles,
                                                      c_sound=c_sound)

    # Monitors to record the states of the simulation for plotting
    spike_mon = SpikeMonitor(mott_neurons) # Records when and which neurons fire
    state_mon = StateMonitor(mott_neurons, 'v', record=True) # Records the time-continuous trace of their voltage

    # Run the physics simulation for 10 real-world milliseconds
    run(10 * ms)

    # ==========================================
    # 3. Analyze Output
    # ==========================================
    # Retrieve the first neuron that successfully fired
    if len(spike_mon.i) > 0:
        predicted_angle = float(target_angles[spike_mon.i[0]] / deg)
    else:
        predicted_angle = None

    if plot_results:
        # --- Plotting Code ---
        plt.figure(figsize=(12, 8))

        # Subplot 1: Membrane Voltages
        plt.subplot(2, 1, 1)

        fired_neurons = set(spike_mon.i) 

        # Plot all neurons' internal voltages
        for i in range(num_neurons):
            angle = int(target_angles[i] / deg)
            if i in fired_neurons:
                # Highlight the winning neuron with a thick green line
                plt.plot(state_mon.t/ms, state_mon.v[i]/mV, color='#2ca02c', linewidth=2.5, zorder=10, label=f'{angle}° (Fired)')
            else:
                # Keep other neurons semi-transparent
                plt.plot(state_mon.t/ms, state_mon.v[i]/mV, color='grey', alpha=0.4, linewidth=1.5, label=f'{angle}°')

        plt.axhline(y=1000, color='k', linestyle='--', label='Threshold (1000 mV)')
        plt.title(f'Internal Excitation of Mott Neurons (True Sound Angle: {true_angle_deg}°)')
        plt.ylabel('Excitation Level (mV)')
        plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left') 
        plt.grid(True)
        plt.xlim(4.5, 6.5) # Zoom onto the time interval where the event happens (around 5ms)

        # Subplot 2: Target Raster Plot
        plt.subplot(2, 1, 2)
        plt.plot(spike_mon.t/ms, spike_mon.i, 'ko', markersize=10)
        plt.yticks(range(num_neurons), [f'{int(ang/deg)}°' for ang in target_angles])
        plt.xlabel('Time (ms)')
        plt.ylabel('Mott Neuron Angle Assignment')
        plt.title('Output Spikes (Localization Result)')
        plt.grid(True, axis='y')
        plt.xlim(4.5, 6.5)

        plt.tight_layout()
        plt.show()

        # Terminal output
        print(f"Sound arrived from {true_angle_deg}°.")
        if predicted_angle is not None:
            print(f"Mott Neuron tuned to {predicted_angle}° successfully fired!")
        else:
            print("No neurons fired. Sound was not localized.")
            
    return predicted_angle

if __name__ == '__main__':
    run_simulation(67, plot_results=True)
