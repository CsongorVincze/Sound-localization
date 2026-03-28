"""
kiindulo_kod.py

This file defines the purely structural algorithms for Leaky Integrate-and-Fire (LIF) 
coincidence detection. It is completely modularized into functions so you can feed in 
ANY microphone array geometry or physics parameters from your other scripts.
"""
from brian2 import *
import numpy as np

def get_default_array_geometry():
    """
    Returns the setup for a standard 4-microphone 10cm x 10cm square physical array.
    """
    mic_x = array([0.05, -0.05, -0.05,  0.05]) * meter
    mic_y = array([0.05,  0.05, -0.05, -0.05]) * meter
    num_mics = len(mic_x)
    return mic_x, mic_y, num_mics

def get_respeaker_array_geometry():
    """
    Returns the real physical geometry of the Seeed ReSpeaker 4-Mic Array (which is physically 
    ~4.65cm wide instead of 10cm).
    """
    # 0.05 meters * 0.465 scale factor = 0.02325 meters (2.3cm from center)
    mic_x = array([0.02325, -0.02325, -0.02325,  0.02325]) * meter
    mic_y = array([0.02325,  0.02325, -0.02325, -0.02325]) * meter
    num_mics = len(mic_x)
    return mic_x, mic_y, num_mics

def get_target_angles(resolution_deg=15):
    """
    Returns an array of target angles in radians, and the degree conversion factor.
    """
    deg = np.pi / 180
    target_angles = np.arange(0, 360, resolution_deg) * deg
    return target_angles, deg

def create_network(mics, 
                   mic_x, 
                   mic_y, 
                   target_angles,
                   c_sound=343 * meter / second,
                   tau_leaky=115 * us,
                   v_thresh=1.0 * volt,
                   v_reset=0.0 * volt,
                   synapse_weight=0.34 * volt,
                   wta_weight=-2.0 * volt):
    """
    Constructs the LIF (Leaky Integrate-and-Fire) Coincidence Detector Network.
    
    Args:
        mics: A Brian2 SpikeGeneratorGroup representing microphone inputs.
        mic_x: A Brian2 array of physical X coordinates for the microphones (e.g. array([...]) * meter)
        mic_y: A Brian2 array of physical Y coordinates for the microphones
        target_angles: A numpy array of angles (in radians) that each neuron should detect.
        c_sound: Physical speed of sound (default: 343 m/s)
        tau_leaky: Determines how fast internal memory (voltage) decays back to 0. (default: 115 us)
        v_thresh: Voltage needed to trigger a spike/detection (default: 1.0 V)
        v_reset: Voltage to return to after a spike happens (default: 0.0 V)
        synapse_weight: Voltage contribution from a single microphone spike (default: 0.34 V)
        wta_weight: Inhibitory voltage applied to all other neurons when one wins (default: -2.0 V)
              
    Returns:
        mott_neurons: The core decision neurons.
        synapses: The connections between the mics and the neurons.
        wta_syns: The Winner-Takes-All inhibitory connections between neurons.
    """
    num_mics = len(mic_x)
    num_neurons = len(target_angles)

    # -------------------------------------------------------------------
    # A. Intrinsic Neuron Dynamics (The Mott Insulator / LIF model)
    # -------------------------------------------------------------------
    mott_eqs = '''
    dv/dt = -v / tau_leaky : volt (unless refractory)
    '''

    mott_neurons = NeuronGroup(num_neurons, mott_eqs, 
                               threshold='v > v_thresh', 
                               reset='v = v_reset', 
                               refractory=2*ms, 
                               method='exact',
                               namespace={'tau_leaky': tau_leaky, 'v_thresh': v_thresh, 'v_reset': v_reset})

    mott_neurons.v = 'rand() * 0.01 * volt'

    # -------------------------------------------------------------------
    # B. Input Connections (Synapses from Mics to Neurons)
    # -------------------------------------------------------------------
    synapses = Synapses(mics, mott_neurons, 
                        'w : volt', 
                        on_pre='v_post += w')
    synapses.connect() 
    synapses.w = synapse_weight 

    # -------------------------------------------------------------------
    # C. Delay Line Tuning (The Core Direction Finding Algorithm)
    # -------------------------------------------------------------------
    for j in range(num_neurons):
        theta_j = target_angles[j]
        
        # Calculate theoretical physical arrival times
        t_acoustic = -(mic_x * cos(theta_j) + mic_y * sin(theta_j)) / c_sound
        t_last = max(t_acoustic) 
        hardware_delays = t_last - t_acoustic
        
        for i in range(num_mics):
            synapses.delay[i, j] = hardware_delays[i]

    # -------------------------------------------------------------------
    # D. Winner-Takes-All Network (WTA)
    # -------------------------------------------------------------------
    # Note: Brian2 string namespaces require variables defined outside to be passed explicitly.
    wta_syns = Synapses(mott_neurons, mott_neurons, 
                        on_pre='v_post += wta_weight',
                        namespace={'wta_weight': wta_weight})
    wta_syns.connect(condition='i != j') 
    wta_syns.delay = 0*us 

    return mott_neurons, synapses, wta_syns