from brian2 import *
import numpy as np
import matplotlib.pyplot as plt

deg = np.pi / 180  # Define the unit for degree conversion

# ==========================================
# 1. Physics & Simulation Setup
# ==========================================
c_sound = 343 * meter / second
# 4 Microphones in a 10cm x 10cm square array
mic_x = array([0.05, -0.05, -0.05,  0.05]) * meter
mic_y = array([0.05,  0.05, -0.05, -0.05]) * meter
num_mics = 4

# Target Angles we want our Mott devices to detect (8 directions)
target_angles = np.arange(0, 360, 15) * deg
num_neurons = len(target_angles)

def run_simulation(true_angle_deg=67, plot_results=True):
    start_scope() # Reset Brian2 simulator
    defaultclock.dt = 1 * us # Smaller time step needed to resolve microsecond coincidences

    # ==========================================
    # 2. Simulate the Acoustic Event
    # ==========================================
    true_angle = true_angle_deg * deg # The actual direction the sound is coming from

    # Calculate acoustic arrival times at each mic
    t_arrival = -(mic_x * cos(true_angle) + mic_y * sin(true_angle)) / c_sound
    t_arrival -= min(t_arrival) # Shift so the first mic gets hit at t=0
    t_arrival += 5 * ms # Add a 5ms baseline delay for simulation buffering

    # Create the Spike Generator (Our "Front-End" Encoder)
    # Emits 1 spike per mic at the exact acoustic arrival time
    indices = array([0, 1, 2, 3])
    mics = SpikeGeneratorGroup(num_mics, indices, t_arrival)

    # ==========================================
    # 3. Define the Mott LIF Neuron
    # ==========================================
    # tau_leaky is tunable via bias voltage (Joule heating). 
    # We set it to 50 microseconds to require strict coincidence.
    tau_leaky = 115 * us
    #!I changed it from 50
    v_thresh = 1.0 * volt 
    v_reset = 0.0 * volt

    # The differential equation: voltage decays over tau_leaky unless refractory
    mott_eqs = '''
    dv/dt = -v / tau_leaky : volt (unless refractory)
    '''

    # Create the reservoir/matrix of Mott neurons
    # Refractory period set to 2ms to prevent multiple spikes per stimulus
    mott_neurons = NeuronGroup(num_neurons, mott_eqs, 
                               threshold='v > v_thresh', 
                               reset='v = v_reset', 
                               refractory=2*ms, 
                               method='exact')

    # Introduce a tiny bit of noise to break perfectly symmetric dead-ties (e.g. at exactly 7.5 degrees)
    mott_neurons.v = 'rand() * 0.01 * volt'

    # ==========================================
    # 4. The Connection Matrix (Hardware Delays)
    # ==========================================
    # Connect all 4 mics to all N Mott neurons
    synapses = Synapses(mics, mott_neurons, 
                        'w : volt', 
                        on_pre='v_post += w')
    synapses.connect() # Full connectivity

    # A single spike adds 0.26 V of excitation. 
    # 4 perfectly timed spikes = 1.04 V (Crosses the 1.0 V threshold)
    # 3 spikes = 0.78 V (Fails to fire)
    synapses.w = 0.34 * volt 
    #!I changed it from 0.26
    # Populate the Delay Matrix
    for j in range(num_neurons):
        theta_j = target_angles[j]
        
        # What are the acoustic delays for a sound coming exactly from theta_j?
        t_acoustic = -(mic_x * cos(theta_j) + mic_y * sin(theta_j)) / c_sound
        
        # The required hardware delay is the difference from the LAST arriving signal
        t_last = max(t_acoustic)
        hardware_delays = t_last - t_acoustic
        
        # Apply these specific physical delays to the synapses connecting to neuron j
        for i in range(num_mics):
            synapses.delay[i, j] = hardware_delays[i]

    # Add Winner-Takes-All (WTA) Lateral Inhibition to enforce EXACTLY 1 spike
    wta_syns = Synapses(mott_neurons, mott_neurons, on_pre='v_post -= 2.0*volt')
    wta_syns.connect(condition='i != j')
    wta_syns.delay = 0*us

    # ==========================================
    # 5. Monitors & Run
    # ==========================================
    spike_mon = SpikeMonitor(mott_neurons)
    # Monitor the internal excitation (voltage) of the neurons
    state_mon = StateMonitor(mott_neurons, 'v', record=True) 

    run(10 * ms)

    # Get predicted angle
    if len(spike_mon.i) > 0:
        predicted_angle = float(target_angles[spike_mon.i[0]] / deg)
    else:
        predicted_angle = None

    if plot_results:
        # ==========================================
        # 6. Plot the Results
        # ==========================================
        plt.figure(figsize=(12, 8))

        # Plot 1: Internal Excitation for all Mott Neurons
        plt.subplot(2, 1, 1)

        fired_neurons = set(spike_mon.i) # Get indices of all neurons that actually fired

        for i in range(num_neurons):
            angle = int(target_angles[i] / deg)
            if i in fired_neurons:
                # Highlight neurons that fired in solid green
                plt.plot(state_mon.t/ms, state_mon.v[i]/mV, color='#2ca02c', linewidth=2.5, zorder=10, label=f'{angle}° (Fired)')
            else:
                # Plot neurons that did not fire semi-transparently
                plt.plot(state_mon.t/ms, state_mon.v[i]/mV, color='grey', alpha=0.4, linewidth=1.5, label=f'{angle}°')

        plt.axhline(y=1000, color='k', linestyle='--', label='Threshold (1000 mV)')
        plt.title(f'Internal Excitation of Mott Neurons (True Sound Angle: {true_angle/deg}°)')
        plt.ylabel('Excitation Level (mV)')
        plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left') # Move legend outside to prevent covering data
        plt.grid(True)
        plt.xlim(4.5, 6.5) # Zoom in on the arrival time

        # Plot 3: Spike Output (Which neuron actually fired?)
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

        # Print text result
        print(f"Sound arrived from {true_angle/deg}°.")
        if predicted_angle is not None:
            print(f"Mott Neuron tuned to {predicted_angle}° successfully fired!")
        else:
            print("No neurons fired. Sound was not localized.")
            
    return predicted_angle

if __name__ == '__main__':
    run_simulation(67, plot_results=True)