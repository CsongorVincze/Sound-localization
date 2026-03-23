import numpy as np
import matplotlib.pyplot as plt
from kiindulo_kod import run_simulation
from brian2 import prefs

# Hide Brian2 compile warnings and reduce console spam during the sweep
import warnings
warnings.filterwarnings('ignore')
prefs.codegen.target = 'numpy'

def sweep_and_plot():
    # Sweep through 0 to 360 degrees (using a step size to save execution time, 
    # but we can do extremely fine if needed, e.g., 2 degrees)
    test_angles = np.arange(0, 360, 2)
    predicted_angles = []

    print(f"Starting sweep from 0 to 360 degrees (Step size: {test_angles[1]-test_angles[0]} degrees)...")
    for i, ang in enumerate(test_angles):
        if i % 10 == 0:
            print(f"Simulated {i}/{len(test_angles)} angles...")
        # We explicitly set plot_results=False so it doesn't pop up 180 windows!
        pred = run_simulation(true_angle_deg=ang, plot_results=False)
        predicted_angles.append(pred)

    predicted_angles = np.array(predicted_angles)
    
    # Filter out None values in case a neuron completely failed to spike
    valid_indices = [i for i, v in enumerate(predicted_angles) if v is not None]
    valid_test = test_angles[valid_indices]
    valid_pred = [predicted_angles[i] for i in valid_indices]

    # Calculate absolute error
    # Note: Angular math error handling (e.g., predicted=0, true=359 -> error is 1, not 359)
    errors = []
    for t, p in zip(valid_test, valid_pred):
        diff = abs(t - p)
        # Wrap the angular error around 180
        if diff > 180:
            diff = 360 - diff
        errors.append(diff)

    # Plotting the Sweep Results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: True vs Predicted Angle
    ax1.plot(test_angles, test_angles, 'k--', label='Ideal Prediction (True = Predicted)')
    ax1.scatter(valid_test, valid_pred, color='#d62728', alpha=0.7, label='SNN Predicted Angle')
    
    # Mark the physical neuron assignment bands
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

    # Plot 2: Absolute Error
    ax2.plot(valid_test, errors, 'o-', color='#1f77b4', markersize=4)
    ax2.text(0.5, 0.9, f"Mean Error: {np.mean(errors):.2f}°", 
             transform=ax2.transAxes, ha='center', fontsize=12,
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
    
    ax2.set_title('Absolute Localization Error over 360° Sweep')
    ax2.set_xlabel('True Angle of Sound (°)')
    ax2.set_ylabel('Absolute Error (°)')
    ax2.set_xlim(0, 360)
    # The max error should logically not exceed half of the 15-deg neuron spacing (7.5°)
    ax2.set_ylim(0, max(errors) + 2 if len(errors) > 0 else 10)
    ax2.grid(True)

    plt.tight_layout()
    # Save the figure locally as well in case the user wants to keep a copy
    plt.savefig('sweep_results.png', dpi=300)
    plt.show()

if __name__ == '__main__':
    sweep_and_plot()
