"""
ReSpeaker v2.0 - Multi-Algorithm DoA Comparison (CORRECTED)

Coordinate System:
- 0° = FRONT (opposite of USB connector)
- 90° = RIGHT
- 180° = BACK (where USB connector is)
- 270° = LEFT
- Angles increase CLOCKWISE

Physical Microphone Layout (looking DOWN at board, USB at bottom):
         FRONT (0°)
           |
    M1 ----+---- M0     (M1=Front-Left, M0=Front-Right)
           |          
   LEFT ---+--- RIGHT
  (270°)   |   (90°)
    M2 ----+---- M3     (M2=Back-Left, M3=Back-Right)
           |
       BACK (180°)
         [USB]
"""
import sounddevice as sd
import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Wedge
import queue
import warnings
warnings.filterwarnings('ignore')

# === CONFIGURATION ===
RATE = 16000
CHUNK_SIZE = 1024
SPEED_OF_SOUND = 343.0

# Distance between adjacent mics (approximate for ReSpeaker v2.0)
MIC_SPACING = 0.0465  # 46.5mm between adjacent mics

# Microphone angles (CLOCKWISE from FRONT = 0°)
# M0 = Front-Right = 45°
# M1 = Front-Left = 315° (NOT 135°!)
# M2 = Back-Left = 225°
# M3 = Back-Right = 135° (NOT 315°!)
MIC_ANGLE_DEG = np.array([45, 315, 225, 135])  # M0, M1, M2, M3

# Distance from center to each mic
MIC_RADIUS = MIC_SPACING / np.sqrt(2)

# Microphone positions in Cartesian (x=RIGHT, y=FRONT)
# x = R * sin(angle), y = R * cos(angle)
MIC_POSITIONS = np.array([
    [MIC_RADIUS * np.sin(np.radians(MIC_ANGLE_DEG[i])), 
     MIC_RADIUS * np.cos(np.radians(MIC_ANGLE_DEG[i]))]
    for i in range(4)
])

audio_queue = queue.Queue()

# ============================================
# DoA ALGORITHMS
# ============================================

def gcc_phat_delay(sig1, sig2, fs, max_tau):
    """
    GCC-PHAT with improvements for narrowband signals.
    Uses modified PHAT weighting to handle pure tones better.
    """
    n = len(sig1) + len(sig2)
    SIG1 = np.fft.rfft(sig1, n=n)
    SIG2 = np.fft.rfft(sig2, n=n)
    
    # Cross-spectrum
    R = SIG1 * np.conj(SIG2)
    
    # Modified PHAT: add a regularization term to avoid division by very small values
    # This helps with narrowband signals
    mag = np.abs(R)
    beta = 0.1 * np.max(mag)  # Regularization
    R = R / (mag + beta)
    
    cc = np.fft.irfft(R, n=n)
    
    max_shift = int(max_tau * fs)
    cc = np.concatenate([cc[-max_shift:], cc[:max_shift+1]])
    
    # Find peak with parabolic interpolation for sub-sample accuracy
    peak_idx = np.argmax(np.abs(cc))
    
    # Parabolic interpolation
    if 1 <= peak_idx <= len(cc) - 2:
        y0, y1, y2 = np.abs(cc[peak_idx-1:peak_idx+2])
        if y0 != 2*y1 - y2:  # Avoid division by zero
            delta = 0.5 * (y0 - y2) / (y0 - 2*y1 + y2)
            peak_idx = peak_idx + delta
    
    tau_samples = peak_idx - max_shift
    return tau_samples / fs

def basic_cc_delay(sig1, sig2, fs, max_tau):
    """Basic cross-correlation without PHAT weighting."""
    n = len(sig1) + len(sig2)
    SIG1 = np.fft.rfft(sig1, n=n)
    SIG2 = np.fft.rfft(sig2, n=n)
    
    R = SIG1 * np.conj(SIG2)
    cc = np.fft.irfft(R, n=n)
    
    max_shift = int(max_tau * fs)
    cc = np.concatenate([cc[-max_shift:], cc[:max_shift+1]])
    
    peak_idx = np.argmax(np.abs(cc))
    
    # Parabolic interpolation
    if 1 <= peak_idx <= len(cc) - 2:
        y0, y1, y2 = np.abs(cc[peak_idx-1:peak_idx+2])
        if y0 != 2*y1 - y2:
            delta = 0.5 * (y0 - y2) / (y0 - 2*y1 + y2)
            peak_idx = peak_idx + delta
    
    tau_samples = peak_idx - max_shift
    return tau_samples / fs

def estimate_doa_gcc_phat(mics, fs):
    """DoA using GCC-PHAT."""
    max_tau = 2 * MIC_SPACING / SPEED_OF_SOUND
    
    # Use opposite mic pairs
    # M0 (45°, front-right) vs M2 (225°, back-left)
    # M1 (315°, front-left) vs M3 (135°, back-right)
    tau_02 = gcc_phat_delay(mics[:, 0], mics[:, 2], fs, max_tau)
    tau_13 = gcc_phat_delay(mics[:, 1], mics[:, 3], fs, max_tau)
    
    return tdoa_to_angle(tau_02, tau_13)

def estimate_doa_basic_cc(mics, fs):
    """DoA using basic cross-correlation."""
    max_tau = 2 * MIC_SPACING / SPEED_OF_SOUND
    
    tau_02 = basic_cc_delay(mics[:, 0], mics[:, 2], fs, max_tau)
    tau_13 = basic_cc_delay(mics[:, 1], mics[:, 3], fs, max_tau)
    
    return tdoa_to_angle(tau_02, tau_13)

def tdoa_to_angle(tau_02, tau_13):
    """
    Convert TDOA from two diagonal mic pairs to angle.
    
    Geometry:
    - M0-M2 diagonal is at 45° to 225° (front-right to back-left)
    - M1-M3 diagonal is at 315° to 135° (front-left to back-right)
    
    For a sound from angle θ (0° = front, clockwise):
    The projection onto M0-M2 axis gives tau_02
    The projection onto M1-M3 axis gives tau_13
    """
    # Maximum possible delay for diagonal pairs
    diag_dist = 2 * MIC_RADIUS
    max_delay = diag_dist / SPEED_OF_SOUND
    
    # Normalize to [-1, 1]
    d02 = np.clip(tau_02 / max_delay, -1, 1)
    d13 = np.clip(tau_13 / max_delay, -1, 1)
    
    # The M0-M2 diagonal is at 45° (measured from front, clockwise)
    # The M1-M3 diagonal is at -45° = 315° (or equivalently, at 135° pointing other way)
    # 
    # For a source at angle θ:
    # d02 = cos(θ - 45°) = cos(θ)cos(45°) + sin(θ)sin(45°)
    # d13 = cos(θ - 315°) = cos(θ - (-45°)) = cos(θ)cos(45°) - sin(θ)sin(45°)
    #
    # Let c = cos(45°) = sin(45°) = 1/√2
    # d02 = c*(cos(θ) + sin(θ))
    # d13 = c*(cos(θ) - sin(θ))
    #
    # Adding: d02 + d13 = 2*c*cos(θ) → cos(θ) = (d02 + d13) / (2*c)
    # Subtracting: d02 - d13 = 2*c*sin(θ) → sin(θ) = (d02 - d13) / (2*c)
    
    c = 1 / np.sqrt(2)
    cos_theta = (d02 + d13) / (2 * c)
    sin_theta = (d02 - d13) / (2 * c)
    
    # Clip to valid range
    cos_theta = np.clip(cos_theta, -1, 1)
    sin_theta = np.clip(sin_theta, -1, 1)
    
    # atan2(sin, cos) gives angle with cos-axis as reference
    angle_rad = np.arctan2(sin_theta, cos_theta)
    angle_deg = np.degrees(angle_rad)
    
    return (angle_deg + 360) % 360

def estimate_doa_srp_phat(mics, fs, resolution=5):
    """SRP-PHAT: Steered Response Power."""
    angles = np.arange(0, 360, resolution)
    powers = np.zeros(len(angles))
    
    n_mics = mics.shape[1]
    n_samples = mics.shape[0]
    
    n_fft = 2 * n_samples
    freqs = np.fft.rfftfreq(n_fft, 1/fs)
    MICs = np.array([np.fft.rfft(mics[:, i], n=n_fft) for i in range(n_mics)])
    
    for idx, angle in enumerate(angles):
        theta = np.radians(angle)
        # Direction vector: x=sin(θ), y=cos(θ) for our convention (0°=front=+y)
        d = np.array([np.sin(theta), np.cos(theta)])
        
        # Delays (negative because sound ARRIVES from this direction)
        delays = -np.dot(MIC_POSITIONS, d) / SPEED_OF_SOUND
        
        steered_sum = np.zeros(len(freqs), dtype=complex)
        for m in range(n_mics):
            phase_shift = np.exp(2j * np.pi * freqs * delays[m])
            mag = np.abs(MICs[m])
            weighted = MICs[m] / (mag + 0.1 * np.max(mag))  # Regularized PHAT
            steered_sum += weighted * phase_shift
        
        powers[idx] = np.sum(np.abs(steered_sum)**2)
    
    return angles[np.argmax(powers)]

def estimate_doa_music(mics, fs, n_sources=1):
    """MUSIC: Multiple Signal Classification."""
    n_mics = mics.shape[1]
    n_samples = mics.shape[0]
    
    R = np.zeros((n_mics, n_mics), dtype=complex)
    
    n_fft = n_samples
    freqs = np.fft.rfftfreq(n_fft, 1/fs)
    MICs = np.array([np.fft.rfft(mics[:, i]) for i in range(n_mics)])
    
    freq_mask = (freqs > 200) & (freqs < 4000)
    
    for f_idx in np.where(freq_mask)[0]:
        x = MICs[:, f_idx].reshape(-1, 1)
        R += x @ x.conj().T
    
    R /= np.sum(freq_mask) + 1e-10
    
    eigenvalues, eigenvectors = eigh(R)
    noise_subspace = eigenvectors[:, :n_mics - n_sources]
    
    angles = np.arange(0, 360, 5)
    spectrum = np.zeros(len(angles))
    
    for idx, angle in enumerate(angles):
        theta = np.radians(angle)
        d = np.array([np.sin(theta), np.cos(theta)])
        
        delays = -np.dot(MIC_POSITIONS, d) / SPEED_OF_SOUND
        center_freq = 1500
        a = np.exp(-2j * np.pi * center_freq * delays)
        a = a.reshape(-1, 1)
        
        denom = np.abs(a.conj().T @ noise_subspace @ noise_subspace.conj().T @ a)
        spectrum[idx] = 1 / (denom[0, 0] + 1e-10)
    
    return angles[np.argmax(spectrum)]

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
        self.history_size = 3
        
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(14, 14))
        self.fig.patch.set_facecolor('#0a0a0a')
        self.fig.suptitle("DoA Algorithm Comparison\n(0°=FRONT, 180°=BACK/USB)", 
                         fontsize=16, color='white', fontweight='bold', y=0.98)
        
        self.axes = []
        self.wedges = []
        self.lines = []
        self.texts = []
        
        for i, (name, _, color) in enumerate(self.algorithms):
            ax = self.fig.add_subplot(2, 2, i+1, projection='polar')
            ax.set_facecolor('#0a0a0a')
            ax.set_title(name, color=color, fontsize=14, fontweight='bold', pad=10)
            
            ax.set_theta_zero_location('N')
            ax.set_theta_direction(-1)
            ax.set_ylim(0, 1.3)
            ax.set_yticks([])
            ax.grid(True, color='#333333', linestyle=':', alpha=0.5)
            ax.set_xticks(np.radians([0, 90, 180, 270]))
            ax.set_xticklabels(['FRONT\n0°', 'RIGHT\n90°', 'BACK\n180°', 'LEFT\n270°'], 
                              color='#888888', fontsize=9)
            
            # Draw mic positions at CORRECT angles
            for j, mic_ang in enumerate(MIC_ANGLE_DEG):
                ax.plot([np.radians(mic_ang)], [0.35], 'o', color='#00ff88', 
                       markersize=7, markeredgecolor='white', markeredgewidth=1)
                ax.text(np.radians(mic_ang), 0.5, f'M{j}', ha='center', va='center',
                       color='#00ff88', fontsize=8)
            
            # Add USB indicator at BACK (180°)
            ax.text(np.radians(180), 1.2, 'USB', ha='center', va='center',
                   color='#ff6666', fontsize=10, fontweight='bold')
            
            wedge = Wedge((0, 0), 1.1, 0, 20, width=0.15,
                         transform=ax.transData._b,
                         facecolor=color, edgecolor='white',
                         linewidth=1, alpha=0.7)
            ax.add_patch(wedge)
            self.wedges.append(wedge)
            
            line, = ax.plot([], [], color=color, linewidth=3)
            self.lines.append(line)
            
            text = ax.text(0.5, 0.5, '0°', transform=ax.transAxes,
                          ha='center', va='center', color=color,
                          fontsize=22, fontweight='bold')
            self.texts.append(text)
            
            self.axes.append(ax)
        
        plt.tight_layout(rect=[0, 0.02, 1, 0.94])
    
    def smooth_angle(self, name, new_angle):
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
        
        while not audio_queue.empty():
            try:
                audio_queue.get_nowait()
            except:
                break
        
        for i, (name, algo_func, _) in enumerate(self.algorithms):
            try:
                angle = algo_func(mics, RATE)
                smooth_angle = self.smooth_angle(name, angle)
            except:
                smooth_angle = 0
            
            self.wedges[i].set_theta1(smooth_angle - 10)
            self.wedges[i].set_theta2(smooth_angle + 10)
            
            angle_rad = np.radians(smooth_angle)
            self.lines[i].set_data([angle_rad, angle_rad], [0, 1.1])
            
            self.texts[i].set_text(f"{smooth_angle:.0f}°")
        
        return self.wedges + self.lines + self.texts

def main():
    print("=" * 60)
    print(" DoA Algorithm Comparison (CORRECTED)")
    print("=" * 60)
    print("""
Coordinate System:
  - 0° = FRONT (opposite USB)
  - 90° = RIGHT  
  - 180° = BACK (where USB connector is)
  - 270° = LEFT

Corrected Microphone Layout:
         FRONT (0°)
           |
    M1 ----.---- M0     (315°)  (45°)
           |          
   LEFT ---+--- RIGHT
  (270°)   |   (90°)
    M2 ----.---- M3     (225°)  (135°)
           |
       BACK (180°)
         [USB]
""")
    
    dev_idx, channels, dev_name = find_respeaker()
    
    if dev_idx is None:
        print("ERROR: ReSpeaker not found!")
        return
    
    print(f"Device: {dev_name}")
    print(f"Channels: {channels}")
    print("\nStarting visualization...")
    
    stream = sd.InputStream(device=dev_idx, channels=channels, samplerate=RATE,
                           blocksize=CHUNK_SIZE, callback=audio_callback,
                           dtype='int16')
    
    with stream:
        viz = MultiAlgorithmVisualizer()
        ani = animation.FuncAnimation(viz.fig, viz.update, interval=80,
                                      blit=False, cache_frame_data=False)
        try:
            plt.show()
        except:
            pass
    
    print("\nStopped.")

if __name__ == "__main__":
    main()
