"""
kiindulo_kod.py

This file defines the purely structural algorithms for Leaky Integrate-and-Fire (LIF) 
coincidence detection. It constructs the physical setup, constants, and the network, 
which can then be imported by other visualization or simulation scripts. 
No simulations are run in this file directly.
"""
from brian2 import *
import numpy as np

# ==========================================
# 1. Physics & Simulation Setup
# ==========================================
deg = np.pi / 180  # Define the unit for degree conversion to radians

# Physical speed of sound in air (meter/second).
# Change this if you want to simulate sound underwater (~1500m/s) or in other media!
c_sound = 343 * meter / second

# 4 Microphones in a 10cm x 10cm square physical array.
# These coordinates map to the physical position of each microphone in meters.
mic_x = array([0.05, -0.05, -0.05,  0.05]) * meter
mic_y = array([0.05,  0.05, -0.05, -0.05]) * meter
num_mics = 4

# Target Angles we want our Mott devices (neurons) to detect.
# We map all 360 degrees into 15-degree resolution chunks (24 directions).
# If you want more precision, lower the step size (e.g., `np.arange(0, 360, 5)`), 
# but keep in mind that hardware limits (tau_leaky, array size) limit real precision.
target_angles = np.arange(0, 360, 15) * deg
num_neurons = len(target_angles) # 24 neurons total

def create_network(mics):
    """
    Constructs the LIF (Leaky Integrate-and-Fire) Coincidence Detector Network.
    
    Args:
        mics: A Brian2 SpikeGeneratorGroup representing microphone inputs.
              These microphones act as the source of our input spikes.
              
    Returns:
        mott_neurons: The core decision neurons.
        synapses: The connections between the mics and the neurons.
        wta_syns: The Winner-Takes-All inhibitory connections between neurons.
    """
    # -------------------------------------------------------------------
    # A. Intrinsic Neuron Dynamics (The Mott Insulator / LIF model)
    # -------------------------------------------------------------------
    # tau_leaky: Determines how fast internal memory (voltage) decays back to 0.
    # It requires spikes to arrive tightly in time to successfully sum up over 1V.
    # If tau_leaky is large, the neuron listens for a longer window (fuzzy precision).
    # If it is small, the neuron requires highly synchronized spikes.
    tau_leaky = 115 * us
    
    # Voltage needed to trigger a spike (detection threshold)
    v_thresh = 1.0 * volt 
    
    # Voltage to return to after a spike happens
    v_reset = 0.0 * volt

    # Differential equation representing decay over time (dv/dt = -v/tau_leaky)
    mott_eqs = '''
    dv/dt = -v / tau_leaky : volt (unless refractory)
    '''

    # The actual group of neurons assigned to each angle
    mott_neurons = NeuronGroup(num_neurons, mott_eqs, 
                               threshold='v > v_thresh', 
                               reset='v = v_reset', 
                               refractory=2*ms, # Prevent spiking continuously
                               method='exact',
                               namespace={'tau_leaky': tau_leaky, 'v_thresh': v_thresh, 'v_reset': v_reset})

    # Small initial random sub-threshold voltage to break perfect symmetry 
    # when an event hits perfectly equidistant paths.
    mott_neurons.v = 'rand() * 0.01 * volt'

    # -------------------------------------------------------------------
    # B. Input Connections (Synapses from Mics to Neurons)
    # -------------------------------------------------------------------
    # Create the full mesh of connections
    synapses = Synapses(mics, mott_neurons, 
                        'w : volt', 
                        on_pre='v_post += w') # Pre-synaptic spike increases Post-synaptic voltage
    synapses.connect() # Every microphone is connected to every neuron

    # The amplitude of a single microphone's voltage contribution.
    # 4 simultaneous spikes will push the voltage to 1.36V (0.34 * 4), crossing the 1.0V threshold.
    # 3 spikes will max out around 1.02V, but because of tau_leaky decay, it usually won't cross 1.0V fast enough.
    synapses.w = 0.34 * volt 

    # -------------------------------------------------------------------
    # C. Delay Line Tuning (The Core Direction Finding Algorithm)
    # -------------------------------------------------------------------
    # Because sound arrives at microphones at different times, we must slow down the 
    # earliest signals using hard-coded delays. By the time the last microphone 
    # receives the sound wave, all delayed earlier signals arrive simultaneously!
    for j in range(num_neurons):
        theta_j = target_angles[j] # This neuron's assigned target angle
        
        # Calculate theoretical physical arrival times based on dot product of wave vector
        t_acoustic = -(mic_x * cos(theta_j) + mic_y * sin(theta_j)) / c_sound
        t_last = max(t_acoustic) # Time the last microphone is hit
        
        # Calculate corrective delay lines: 
        # (Last mic gets 0 delay, earliest mic gets maximum delay)
        hardware_delays = t_last - t_acoustic
        
        # Apply the specific hardware delays to the neuronal path
        for i in range(num_mics):
            synapses.delay[i, j] = hardware_delays[i]

    # -------------------------------------------------------------------
    # D. Winner-Takes-All Network (WTA)
    # -------------------------------------------------------------------
    # When one neuron fires (detects a match), it instantly suppresses (-2.0V) 
    # all other neurons. This prevents adjacent neurons (e.g. 45 deg and 60 deg) 
    # from firing due to slight leakage overlaps, cleaning up the final decision 
    # and forcing exactly one neuron to "win".
    wta_syns = Synapses(mott_neurons, mott_neurons, on_pre='v_post -= 2.0*volt')
    wta_syns.connect(condition='i != j') # Connect to everyone except yourself
    wta_syns.delay = 0*us # Inhibition applies instantaneously, guaranteeing competition

    return mott_neurons, synapses, wta_syns