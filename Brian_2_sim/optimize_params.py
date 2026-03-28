"""
optimize_params.py

A standalone script to execute a theoretical computational grid search over the 
SNN parameters. This will find the absolute lowest error limits for the given
hardware microphone layout.
"""
import numpy as np
import matplotlib.pyplot as plt
from brian2 import prefs, us, volt
import warnings

# Import the base simulation runner
from ideal_sim import run_simulation

def optimize_parameters():
    """
    Performs a theoretical sweep over different combinations of tau_leaky and v_thresh
    to find the optimal settings that minimize localization error.
    """
    warnings.filterwarnings('ignore')
    prefs.codegen.target = 'numpy'

    print("Starting theoretical parameter optimization grid search...")
    
    # Define the 10x10 Grid over parameters
    tau_leaks = np.linspace(40, 200, 10)  # Sweep decay window from 40 to 200 microseconds
    v_threshs = np.linspace(0.4, 1.3, 10) # Sweep voltage threshold from 0.4V to 1.3V (w is 0.34V)
    
    # We test on angles exactly between our 15-degree neurons (the hardest edge cases) 
    # plus exact matches ranging from 0 to 90 degrees to get a complete mapping average.
    test_angles = np.arange(0, 90, 7.5) 
    
    error_matrix = np.zeros((len(v_threshs), len(tau_leaks)))
    success_matrix = np.zeros((len(v_threshs), len(tau_leaks)))
    
    best_error = float('inf')
    best_params = (None, None)
    
    total_runs = len(tau_leaks) * len(v_threshs) * len(test_angles)
    print(f"Total theoretical simulations to run: {total_runs}")
    print(f"Testing {len(test_angles)} angles per physics setup...")
    
    run_count = 0
    for i, vt in enumerate(v_threshs):
        for j, tl in enumerate(tau_leaks):
            errors = []
            successes = 0
            
            for ang in test_angles:
                run_count += 1
                if run_count % 100 == 0:
                    print(f"Progress: {run_count}/{total_runs} simulations...")
                    
                # We suppress plotting by passing plot_results=False    
                pred = run_simulation(true_angle_deg=ang, 
                                      plot_results=False, 
                                      tau_leaky_val=tl * us, 
                                      v_thresh_val=vt * volt)
                
                if pred is not None:
                    diff = abs(ang - pred)
                    if diff > 180:
                        diff = 360 - diff
                    errors.append(diff)
                    successes += 1
                else:
                    errors.append(90) # Massive penalty if the network failed to fire completely!
                    
            mean_error = np.mean(errors)
            error_matrix[i, j] = mean_error
            success_matrix[i, j] = successes / len(test_angles)
            
            if mean_error < best_error:
                best_error = mean_error
                best_params = (vt, tl)

    print(f"\nOptimization Finished!")
    print(f"Optimal v_thresh:  {best_params[0]:.2f} V")
    print(f"Optimal tau_leaky: {best_params[1]:.1f} us")
    print(f"Minimum Avg Error: {best_error:.2f}°")
    
    # Save the exact matrices and optimal results to a local text file!
    with open('optimization_results.txt', 'w') as f:
        f.write("=========================================\n")
        f.write("       OPTIMIZATION RESULTS SUMMARY      \n")
        f.write("=========================================\n")
        f.write(f"Optimal v_thresh:  {best_params[0]:.2f} V\n")
        f.write(f"Optimal tau_leaky: {best_params[1]:.1f} us\n")
        f.write(f"Minimum Avg Error: {best_error:.2f}°\n\n")
        
        f.write("--- V_THRESH SWEEP RANGE (V) ---\n")
        f.write(", ".join([f"{v:.2f}" for v in v_threshs]) + "\n\n")
        
        f.write("--- TAU_LEAKY SWEEP RANGE (us) ---\n")
        f.write(", ".join([f"{t:.1f}" for t in tau_leaks]) + "\n\n")

        f.write("=========================================\n")
        f.write("  ERROR MATRIX (Mean Absolute Error in °)\n")
        f.write("=========================================\n")
        f.write("Rows correspond to v_thresh, Columns correspond to tau_leaky\n")
        np.savetxt(f, error_matrix, fmt="%6.2f", delimiter=", ")
        f.write("\n")
        
        f.write("=========================================\n")
        f.write(" SUCCESS MATRIX (Firing Reliability 0-1) \n")
        f.write("=========================================\n")
        f.write("Rows correspond to v_thresh, Columns correspond to tau_leaky\n")
        np.savetxt(f, success_matrix, fmt="%6.2f", delimiter=", ")
    
    # ==========================================
    # Plotting Optimization Results
    # ==========================================
    plt.figure(figsize=(14, 6))
    
    # Heatmap 1: The mean error across all angles tested
    plt.subplot(1, 2, 1)
    im1 = plt.imshow(error_matrix, origin='lower', aspect='auto', cmap='viridis_r',
                     extent=[tau_leaks[0], tau_leaks[-1], v_threshs[0], v_threshs[-1]],
                     vmin=np.min(error_matrix), vmax=15)
    plt.colorbar(im1, label='Mean Absolute Error (°)', extend='max')
    plt.xlabel('tau_leaky (us)')
    plt.ylabel('v_thresh (V)')
    plt.title('Theoretical Localization Error\n(Lighter/Yellow is Better)')
    plt.plot(best_params[1], best_params[0], 'r*', markersize=15, label='Optimal')
    plt.legend()
    
    # Heatmap 2: Did the network successfully detect anything at all?
    plt.subplot(1, 2, 2)
    im2 = plt.imshow(success_matrix, origin='lower', aspect='auto', cmap='plasma',
                     extent=[tau_leaks[0], tau_leaks[-1], v_threshs[0], v_threshs[-1]])
    plt.colorbar(im2, label='Success Rate (1.0 = All fired)')
    plt.xlabel('tau_leaky (us)')
    plt.ylabel('v_thresh (V)')
    plt.title('Network Firing Reliability\n(Yellow = Fired safely, Blue = Silent network)')
    plt.plot(best_params[1], best_params[0], 'w*', markersize=15, label='Optimal')
    
    plt.tight_layout()
    plt.savefig('optimization_results.png', dpi=300)
    plt.show()

if __name__ == '__main__':
    optimize_parameters()
