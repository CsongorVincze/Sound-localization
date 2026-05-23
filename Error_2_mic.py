import numpy as np
import matplotlib.pyplot as plt

def plot_theoretical_error():
    # --- System Parameters ---
    c = 343.0      # Speed of sound (m/s)
    d = 0.20       # Distance between microphones (meters)
    
    # We use a tiny time error (delta_tau) to scale the derivative into degrees.
    # Here, we use ~20.8 microseconds (the duration of 1 sample at 48kHz)
    delta_tau = 1.0 / 48000.0 

    # --- 1. Define True Angles ---
    # -89 to 89 to avoid dividing by exactly zero at cos(+/-90)
    theta_true_deg = np.linspace(-89, 89, 1000)
    theta_true_rad = np.radians(theta_true_deg)

    # --- 2. Apply the Derivative Formula ---
    # d(theta)/d(tau) = c / (d * cos(theta)) -> Result is in Radians per Second
    derivative_rad_per_sec = c / (d * np.cos(theta_true_rad))

    # --- 3. Calculate Theoretical Angular Error ---
    # delta_theta = derivative * delta_tau
    error_rad = derivative_rad_per_sec * delta_tau
    error_deg = np.degrees(error_rad)

    # --- 4. Plotting ---
    plt.figure(figsize=(10, 6))
    plt.plot(theta_true_deg, error_deg, linewidth=2.5, color='#ff7f0e', label=r'Elméleti hiba ($\Delta\theta$)')

    # Formatting the plot
    plt.title('Két mikrofonos rendszer elméleti hibája', fontsize=14, pad=15)
    plt.xlabel('Valódi érkezési irány (Fok) [0° = Oldirány]', fontsize=12)
    plt.ylabel(f'Szöghiba $\Delta\\tau$ = {delta_tau*1e6:.1f} µs (Fok)', fontsize=12)
    
    # Grid and markers
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.axvline(x=0, color='blue', linestyle='--', alpha=0.5, label='Oldirány (Maximális pontosság)')
    
    plt.xlim(-90, 90)
    
    # Set y-limit to zoom in on the usable curve, ignoring the infinity spike at exactly 90
    plt.ylim(0, np.max(error_deg[100:900]) * 3) 
    
    plt.legend(loc='upper center', fontsize=11)
    plt.tight_layout()
    plt.savefig("ket_mikrofon_elmeleti_hiba")
    plt.show()

if __name__ == "__main__":
    plot_theoretical_error()