# Spiking Neural Network (SNN) Direction of Arrival (DoA) Architecture

This document explains the topology, governing mathematics, and physical parameters of the Mott-neuron-inspired acoustic localization algorithm.

## 1. High-Level Concept

The network is designed to solve **Direction of Arrival (DoA)** for a sound wave hitting a physical array of microphones. 
Because sound travels at a finite speed ($c = 343$ m/s), a sound wave originating from a specific angle will hit earlier microphones before reaching further ones. The SNN captures this spatial phase difference and translates it into precisely orchestrated **temporal coincidences** that trigger exactly one output neuron.

---

## 2. Network Topology

The network consists of three distinct functional layers/mechanisms:

### A. The Input Layer (Microphone Encoders)
4 sensors arranged in a 10cm x 10cm square. 
They act as **Spike Generators**. When the acoustic wave washes over the array, each microphone emits exactly **1 spike** at the exact microsecond the sound hits it.
```python
mic_x = array([0.05, -0.05, -0.05,  0.05]) * meter
mic_y = array([0.05,  0.05, -0.05, -0.05]) * meter

# Emits 1 spike per mic at the acoustic arrival time
mics = SpikeGeneratorGroup(num_mics, indices, t_arrival)
```

### B. The Processing Layer (Mott LIF Neurons)
We define 24 **Leaky Integrate-and-Fire (LIF)** neurons. Each neuron acts as a dedicated "listener" exclusively tuned to a 15° slice of the 360° environment (0°, 15°, 30°, etc.). 

If multiple spikes arrive at a neuron at the *exact same microsecond*, their voltages stack up, pushing the neuron past its firing threshold. If they arrive staggered, the voltage inherently "leaks" away between arrivals, failing to trigger a spike.
```python
# The differential equation: voltage decays over tau_leaky
mott_eqs = '''
dv/dt = -v / tau_leaky : volt (unless refractory)
'''
mott_neurons = NeuronGroup(num_neurons, mott_eqs, 
                           threshold='v > v_thresh', 
                           reset='v = v_reset', 
                           refractory=2*ms, 
                           method='exact')
```

### C. The Delay Synapses (Hardware Compensators)
This is where the magic happens. Every microphone is connected to every neuron. However, each synaptic connection has a meticulously calculated **hardware delay**.
If a neuron is tuned to 45°, its incoming synapses intentionally delay the signals from the *first* microphones hit so that they patiently "wait" for the signals from the *last* microphones. 
```python
# Apply these specific physical delays to the synapses connecting to neuron j
for i in range(num_mics):
    synapses.delay[i, j] = hardware_delays[i]
```
If the sound actually came from 45°, all 4 delayed spikes will perfectly align and hit the neuron in unison.

### D. Winner-Takes-All (WTA) Lateral Inhibition
Because an angle like `67°` falls in between the `60°` and `75°` neurons, the staggered spikes might accidentally stack enough voltage in *both* neurons to trigger them concurrently.
To force the network into a strict **1-of-24** classification, we fully interconnect the 24 neurons with intense **lateral inhibition**. The very first fractional microsecond a single neuron hits the threshold, it fires a `-2.0V` penalty to all peers, permanently shutting down their ability to fire during that sound event.
```python
wta_syns = Synapses(mott_neurons, mott_neurons, on_pre='v_post -= 2.0*volt')
wta_syns.connect(condition='i != j') # Connect to everyone except itself
wta_syns.delay = 0*us
```

---

## 3. Core Tuning Parameters

The physical parameters dictate strictness:

*   **`tau_leaky = 115 * us`**
    This determines how fast the internal voltage drops back to 0. A massive tau (e.g. `1000us`) would mean spikes from wildly wrong angles would still stack up and fire. A tiny tau (e.g. `10us`) means the 4 spikes must arrive in mathematically perfect unison. `115us` strikes the perfect balance, allowing "close enough" off-grid angles (like 67°) to stack successfully.
    
*   **`synapses.w = 0.34 * volt`** & **`v_thresh = 1.0 * volt`**
    Each spike adds `0.34V`. 
    - 4 perfectly timed spikes sum to `1.36V` (easily crossing the `1.0V` threshold).
    - 3 perfectly timed spikes sum to `1.02V` (barely triggering).
    - 2 perfectly timed spikes sum to `0.68V` (fails to fire).
    Because of the leak, an imperfectly timed set of 4 spikes might max out at `1.09V`, ensuring at least one neuron reliably fires over the 1.0V line.
    
*   **`refractory = 2 * ms`**
    Once a neuron spikes, it is completely locked down for $2$ milliseconds. This reliably prevents the winner neuron from "double-spiking" while the lingering acoustic tail finishes propagating through the delayed synapses.
    
*   **`mott_neurons.v = 'rand() * 0.01 * volt'`**
    A microscopic random voltage baseline injected into the initial state. In rare, mathematically perfect mid-points (like a sound exactly at 7.5°), the 0° and 15° neurons would cross the threshold in the exact same `0.000` float step, causing Brian2 to fire both simultaneously despite the inhibition. This imperceptible noise breaks the algorithmic tie.
