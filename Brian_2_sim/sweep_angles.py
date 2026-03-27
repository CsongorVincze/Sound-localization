"""
sweep_angles.py

This script systematically tests the localization accuracy of the algorithm over 
the full 360-degree range. It relies on the ideal acoustic simulation to measure
hardware baseline performance and plots a comprehensive error report.
"""
import numpy as np
import matplotlib.subplots as plt_sub
import matplotlib.pyplot as plt
# We import run_simulation from our newly split ideal_sim file, instead of kiindulo_kod
from ideal_sim import run_simulation
from brian2 import prefs

# Hide Brian2 compile warnings and reduce console spam during the sweep.
# The code generation target is set to numpy for simpler, faster, bug-free execution.
import warnings
warnings.filterwarnings('ignore')
prefs.codegen.target = 'numpy'

def sweep_and_plot():
    """
    Sweeps through every angle from 0 to 360 in small steps (e.g. 2 degrees).
    The error indicates the network's inherent mapping limits and helps us tune `tau_leaky`.
    """
    # Step size of 2 degrees gives a high-resolution sweep of the network's behavior.
    test_angles = np.arange(0, 360, 2)
    predicted_angles = []

    print(f"Starting sweep from 0 to 360 degrees (Step size: {test_angles[1]-test_angles[0]} degrees)...")
    
    # Run a full simulation for every angle
    for i, ang in enumerate(test_angles):
        if i % 10 == 0:
            print(f"Simulated {i}/{len(test_angles)} angles...")
            
        # We explicitly set plot_results=False so we don't spam 180 individual result windows!
        # This purely retrieves the algorithm's guess for the specific physical angle.
        pred = run_simulation(true_angle_deg=ang, plot_results=False)
        predicted_angles.append(pred)

    predicted_angles = np.array(predicted_angles)
    
    # Filter out None values in case a neuron completely failed to spike.
    # A None value means the threshold voltage (1.0V) was never crossed despite acoustic input.
    valid_indices = [i for i, v in enumerate(predicted_angles) if v is not None]
    valid_test = test_angles[valid_indices]
    valid_pred = [predicted_angles[i] for i in valid_indices]

    # Calculate absolute error
    errors = []
    for t, p in zip(valid_test, valid_pred):
        diff = abs(t - p)
        # Wrap the angular error around 180 degrees.
        # e.g., predicted=0, true=359 -> error is 1, not 359!
        if diff > 180:
            diff = 360 - diff
        errors.append(diff)

    # ==========================================
    # Plotting the Sweep Results
    # ==========================================
    # We display two plots: 
    # 1. The scatter mapping of True vs. Predicted
    # 2. The Absolute Error for specific angles
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # --- Plot 1: True vs Predicted Angle ---
    ax1.plot(test_angles, test_angles, 'k--', label='Ideal Prediction (True = Predicted)')
    ax1.scatter(valid_test, valid_pred, color='#d62728', alpha=0.7, label='SNN Predicted Angle')
    
    # Mark the physical neuron assignment bands.
    # Neurons are spaced 15 degrees apart; this produces horizontal "steps" indicating quantization!
    neuron_resolutions = np.arange(0, 360, 15)
    for res in neuron_resolutions:
        ax1.axhline(y=res, color='gray', linestyle=':', alpha=0.3)
        
    ax1.set_title('WTA SNN Localization Sweep')
    ax1.set_xlabel('True Angle of Sound (°)')
    ax1.set_ylabel('Predicted Mott Neuron Output (°)')
    ax1.set_xlim(0, 360)
    ax1.set_ylim(0, 360)
    ax1.legend()
    ax1.grid(True)

    # --- Plot 2: Absolute Error ---
    ax2.plot(valid_test, errors, 'o-', color='#1f77b4', markersize=4)
    # Show the average error on the screen.
    ax2.text(0.5, 0.9, f"Mean Error: {np.mean(errors):.2f}°", 
             transform=ax2.transAxes, ha='center', fontsize=12,
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
    
    ax2.set_title('Absolute Localization Error over 360° Sweep')
    ax2.set_xlabel('True Angle of Sound (°)')
    ax2.set_ylabel('Absolute Error (°)')
    ax2.set_xlim(0, 360)
    
    # The maximum logical quantization error should ideally not exceed half 
    # of the 15-deg neuron spacing (7.5°). If it does, the WTA might be misfiring.
    ax2.set_ylim(0, max(errors) + 2 if len(errors) > 0 else 10)
    ax2.grid(True)

    plt.tight_layout()
    # Save the figure locally as well in case the user wants to keep a copy
    plt.savefig('sweep_results.png', dpi=300)
    plt.show()

if __name__ == '__main__':
    sweep_and_plot()
