"""
ReSpeaker v2.0 - Multi-Algorithm DoA Comparison
Compares different Direction of Arrival algorithms in real-time:
1. GCC-PHAT - Generalized Cross-Correlation with Phase Transform
2. SRP-PHAT - Steered Response Power with PHAT
3. MUSIC - Multiple Signal Classification
4. Basic Cross-Correlation
"""
import sounddevice as sd
import numpy as np
from scipy import signal
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Wedge, Circle
import queue
import warnings
warnings.filterwarnings('ignore')

# === CONFIGURATION ===
RATE = 16000
CHUNK_SIZE = 1024
SPEED_OF_SOUND = 343.0
MIC_DISTANCE = 0.058
MIC_RADIUS = MIC_DISTANCE / 2

# Microphone positions (x, y) in meters - circular array
MIC_POSITIONS = np.array([
    [MIC_RADIUS, 0],           # Mic 0: Right (0°)
    [0, MIC_RADIUS],           # Mic 1: Front (90°)
    [-MIC_RADIUS, 0],          # Mic 2: Left (180°)
    [0, -MIC_RADIUS]           # Mic 3: Back (270°)
])

audio_queue = queue.Queue()

# ============================================
# DoA ALGORITHMS
# ============================================

def gcc_phat(sig1, sig2, fs, max_tau=None):
    """GCC-PHAT: Good for reverberant environments, robust to noise."""
    n = len(sig1) + len(sig2)
    SIG1 = np.fft.rfft(sig1, n=n)
    SIG2 = np.fft.rfft(sig2, n=n)
    R = SIG1 * np.conj(SIG2)
    R = R / (np.abs(R) + 1e-10)  # PHAT weighting
    cc = np.fft.irfft(R, n=n)
    
    max_shift = int((max_tau or 0.001) * fs)
    cc = np.concatenate([cc[-max_shift:], cc[:max_shift+1]])
    tau_samples = np.argmax(np.abs(cc)) - max_shift
    return tau_samples / fs

def basic_cross_correlation(sig1, sig2, fs, max_tau=None):
    """Basic CC: Simple, fast, but sensitive to noise and reverb."""
    n = len(sig1) + len(sig2)
    SIG1 = np.fft.rfft(sig1, n=n)
    SIG2 = np.fft.rfft(sig2, n=n)
    R = SIG1 * np.conj(SIG2)
    # No PHAT weighting - just raw cross-correlation
    cc = np.fft.irfft(R, n=n)
    
    max_shift = int((max_tau or 0.001) * fs)
    cc = np.concatenate([cc[-max_shift:], cc[:max_shift+1]])
    tau_samples = np.argmax(np.abs(cc)) - max_shift
    return tau_samples / fs

def doa_from_tdoa(tau_02, tau_13, mic_dist):
    """Convert TDOA to angle using pair geometry."""
    max_delay = mic_dist / SPEED_OF_SOUND
    sin_theta1 = np.clip(tau_02 / max_delay, -1, 1)
    sin_theta2 = np.clip(tau_13 / max_delay, -1, 1)
    angle_rad = np.arctan2(sin_theta1, sin_theta2)
    angle_deg = np.degrees(angle_rad)
    return (angle_deg + 360) % 360

def estimate_doa_gcc_phat(mics, fs):
    """DoA using GCC-PHAT on opposite mic pairs."""
    max_tau = MIC_DISTANCE / SPEED_OF_SOUND * 2
    tau_02 = gcc_phat(mics[:, 0], mics[:, 2], fs, max_tau)
    tau_13 = gcc_phat(mics[:, 1], mics[:, 3], fs, max_tau)
    return doa_from_tdoa(tau_02, tau_13, MIC_DISTANCE)

def estimate_doa_basic_cc(mics, fs):
    """DoA using basic cross-correlation."""
    max_tau = MIC_DISTANCE / SPEED_OF_SOUND * 2
    tau_02 = basic_cross_correlation(mics[:, 0], mics[:, 2], fs, max_tau)
    tau_13 = basic_cross_correlation(mics[:, 1], mics[:, 3], fs, max_tau)
    return doa_from_tdoa(tau_02, tau_13, MIC_DISTANCE)

def estimate_doa_srp_phat(mics, fs, resolution=5):
    """
    SRP-PHAT: Steered Response Power with PHAT weighting.
    Scans all directions and finds the one with maximum power.
    More robust but computationally heavier.
    """
    angles = np.arange(0, 360, resolution)
    powers = np.zeros(len(angles))
    
    n_mics = mics.shape[1]
    n_samples = mics.shape[0]
    
    # FFT of all channels
    n_fft = 2 * n_samples
    freqs = np.fft.rfftfreq(n_fft, 1/fs)
    MICs = np.array([np.fft.rfft(mics[:, i], n=n_fft) for i in range(n_mics)])
    
    for idx, angle in enumerate(angles):
        # Direction vector
        theta = np.radians(angle)
        d = np.array([np.cos(theta), np.sin(theta)])
        
        # Calculate expected delays for each mic
        delays = np.dot(MIC_POSITIONS, d) / SPEED_OF_SOUND
        
        # Steer and sum
        steered_sum = np.zeros(len(freqs), dtype=complex)
        for m in range(n_mics):
            phase_shift = np.exp(2j * np.pi * freqs * delays[m])
            # PHAT weighting
            weighted = MICs[m] / (np.abs(MICs[m]) + 1e-10)
            steered_sum += weighted * phase_shift
        
        # Power
        powers[idx] = np.sum(np.abs(steered_sum)**2)
    
    best_idx = np.argmax(powers)
    return angles[best_idx]

def estimate_doa_music(mics, fs, n_sources=1):
    """
    MUSIC: MUltiple SIgnal Classification.
    High resolution, works well for multiple sources.
    Uses eigenvalue decomposition of the covariance matrix.
    """
    n_mics = mics.shape[1]
    n_samples = mics.shape[0]
    
    # Build covariance matrix
    R = np.zeros((n_mics, n_mics), dtype=complex)
    
    # Use frequency-domain covariance
    n_fft = n_samples
    freqs = np.fft.rfftfreq(n_fft, 1/fs)
    MICs = np.array([np.fft.rfft(mics[:, i]) for i in range(n_mics)])
    
    # Select frequency bins with energy
    freq_mask = (freqs > 100) & (freqs < 4000)  # Voice frequency range
    
    for f_idx in np.where(freq_mask)[0]:
        x = MICs[:, f_idx].reshape(-1, 1)
        R += x @ x.conj().T
    
    R /= np.sum(freq_mask)
    
    # Eigendecomposition
    eigenvalues, eigenvectors = eigh(R)
    
    # Noise subspace (smallest eigenvalues)
    noise_subspace = eigenvectors[:, :n_mics - n_sources]
    
    # Scan angles
    angles = np.arange(0, 360, 5)
    spectrum = np.zeros(len(angles))
    
    for idx, angle in enumerate(angles):
        theta = np.radians(angle)
        d = np.array([np.cos(theta), np.sin(theta)])
        
        # Steering vector (simplified - assume narrowband)
        delays = np.dot(MIC_POSITIONS, d) / SPEED_OF_SOUND
        center_freq = 1000  # Use 1kHz as reference
        a = np.exp(-2j * np.pi * center_freq * delays)
        a = a.reshape(-1, 1)
        
        # MUSIC spectrum
        denom = np.abs(a.conj().T @ noise_subspace @ noise_subspace.conj().T @ a)
        spectrum[idx] = 1 / (denom[0, 0] + 1e-10)
    
    best_idx = np.argmax(spectrum)
    return angles[best_idx]

# ============================================
# AUDIO HANDLING
# ============================================

def audio_callback(indata, frames, time_info, status):
    audio_queue.put(indata.copy())

def find_respeaker():
    for i, dev in enumerate(sd.query_devices()):
        if dev['max_input_channels'] > 0:
            name = dev['name'].lower()
            if 'respeaker' in name or 'uac1.0' in name:
                return i, dev['max_input_channels'], dev['name']
    return None, None, None

def get_mics(audio_chunk):
    """Extract 4 microphone channels."""
    n_ch = audio_chunk.shape[1]
    if n_ch >= 6:
        return audio_chunk[:, 1:5].astype(np.float64)
    elif n_ch >= 4:
        return audio_chunk[:, 0:4].astype(np.float64)
    return None

# ============================================
# VISUALIZATION
# ============================================

class MultiAlgorithmVisualizer:
    def __init__(self):
        self.algorithms = [
            ("GCC-PHAT", estimate_doa_gcc_phat, "#00d2ff"),
            ("Basic CC", estimate_doa_basic_cc, "#ff6b6b"),
            ("SRP-PHAT", estimate_doa_srp_phat, "#4ecdc4"),
            ("MUSIC", estimate_doa_music, "#ffe66d"),
        ]
        
        self.angle_history = {name: [] for name, _, _ in self.algorithms}
        self.history_size = 5
        
        # Create figure with 4 subplots
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(16, 16))
        self.fig.patch.set_facecolor('#0a0a0a')
        self.fig.suptitle("DoA Algorithm Comparison", fontsize=20, 
                         color='white', fontweight='bold', y=0.98)
        
        self.axes = []
        self.wedges = []
        self.lines = []
        self.texts = []
        
        for i, (name, _, color) in enumerate(self.algorithms):
            ax = self.fig.add_subplot(2, 2, i+1, projection='polar')
            ax.set_facecolor('#0a0a0a')
            ax.set_title(name, color=color, fontsize=16, fontweight='bold', pad=15)
            
            # Configure polar
            ax.set_theta_zero_location('N')
            ax.set_theta_direction(-1)
            ax.set_ylim(0, 1.3)
            ax.set_yticks([])
            ax.grid(True, color='#333333', linestyle=':', alpha=0.5)
            ax.set_xticks(np.radians([0, 45, 90, 135, 180, 225, 270, 315]))
            ax.set_xticklabels(['F', '', 'R', '', 'B', '', 'L', ''], 
                              color='#666666', fontsize=10)
            
            # Draw mic positions
            for j in range(4):
                mic_angle = np.radians(j * 90)  # 0, 90, 180, 270
                ax.plot([mic_angle], [0.3], 'o', color='#00ff88', 
                       markersize=8, markeredgecolor='white', markeredgewidth=1)
            
            # Wedge
            wedge = Wedge((0, 0), 1.1, 0, 30, width=0.2,
                         transform=ax.transData._b,
                         facecolor=color, edgecolor='white',
                         linewidth=1, alpha=0.6)
            ax.add_patch(wedge)
            self.wedges.append(wedge)
            
            # Direction line
            line, = ax.plot([], [], color=color, linewidth=3)
            self.lines.append(line)
            
            # Angle text
            text = ax.text(0.5, 0.5, '0°', transform=ax.transAxes,
                          ha='center', va='center', color=color,
                          fontsize=24, fontweight='bold')
            self.texts.append(text)
            
            self.axes.append(ax)
        
        plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    
    def smooth_angle(self, name, new_angle):
        """Circular averaging for smooth display."""
        history = self.angle_history[name]
        history.append(new_angle)
        if len(history) > self.history_size:
            history.pop(0)
        
        angles_rad = np.radians(history)
        mean_sin = np.mean(np.sin(angles_rad))
        mean_cos = np.mean(np.cos(angles_rad))
        smooth = np.degrees(np.arctan2(mean_sin, mean_cos))
        return (smooth + 360) % 360
    
    def update(self, frame):
        if audio_queue.empty():
            return self.wedges + self.lines + self.texts
        
        data = audio_queue.get()
        mics = get_mics(data)
        
        if mics is None:
            return self.wedges + self.lines + self.texts
        
        # Clear queue to prevent lag
        while not audio_queue.empty():
            try:
                audio_queue.get_nowait()
            except:
                break
        
        # Calculate DoA with each algorithm
        for i, (name, algo_func, color) in enumerate(self.algorithms):
            try:
                angle = algo_func(mics, RATE)
                smooth_angle = self.smooth_angle(name, angle)
            except Exception as e:
                smooth_angle = 0
            
            # Update wedge
            self.wedges[i].set_theta1(smooth_angle - 15)
            self.wedges[i].set_theta2(smooth_angle + 15)
            
            # Update line
            angle_rad = np.radians(smooth_angle)
            self.lines[i].set_data([angle_rad, angle_rad], [0, 1.1])
            
            # Update text
            self.texts[i].set_text(f"{smooth_angle:.0f}°")
        
        return self.wedges + self.lines + self.texts

def main():
    print("=" * 60)
    print(" DoA Algorithm Comparison")
    print("=" * 60)
    print("""
Comparing 4 Direction of Arrival algorithms:

1. GCC-PHAT (Cyan)
   - Generalized Cross-Correlation with Phase Transform
   - Best for: Reverberant environments, robust to multipath
   
2. Basic Cross-Correlation (Red)
   - Simple time-domain correlation
   - Best for: Low noise, anechoic conditions
   
3. SRP-PHAT (Teal)
   - Steered Response Power with PHAT weighting
   - Best for: Multiple sources, high accuracy
   
4. MUSIC (Yellow)
   - Multiple Signal Classification (subspace method)
   - Best for: High resolution, multiple sources
""")
    
    # Find device
    dev_idx, channels, dev_name = find_respeaker()
    
    if dev_idx is None:
        print("ERROR: ReSpeaker not found!")
        return
    
    print(f"Device: {dev_name}")
    print(f"Channels: {channels}")
    print("\nStarting visualization...")
    print("Make sounds from different directions!")
    
    # Start audio
    stream = sd.InputStream(device=dev_idx, channels=channels, samplerate=RATE,
                           blocksize=CHUNK_SIZE, callback=audio_callback,
                           dtype='int16')
    
    with stream:
        viz = MultiAlgorithmVisualizer()
        ani = animation.FuncAnimation(viz.fig, viz.update, interval=100,
                                      blit=False, cache_frame_data=False)
        try:
            plt.show()
        except:
            pass
    
    print("\nStopped.")

if __name__ == "__main__":
    main()
