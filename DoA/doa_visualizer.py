"""
ReSpeaker v2.0 - DoA Visualization
Real-time polar compass showing direction of sound source
"""
import sounddevice as sd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Wedge
import queue
import sys

# === CONFIGURATION ===
RATE = 16000
CHUNK_SIZE = 1024
SPEED_OF_SOUND = 343.0
MIC_DISTANCE = 0.058
MIC_RADIUS = MIC_DISTANCE / 2
MIC_ANGLES = np.array([0, np.pi/2, np.pi, 3*np.pi/2])

audio_queue = queue.Queue()

def gcc_phat(sig1, sig2, fs, max_tau=None):
    """GCC-PHAT algorithm for time delay estimation."""
    n = len(sig1) + len(sig2)
    SIG1 = np.fft.rfft(sig1, n=n)
    SIG2 = np.fft.rfft(sig2, n=n)
    R = SIG1 * np.conj(SIG2)
    R = R / (np.abs(R) + 1e-10)
    cc = np.fft.irfft(R, n=n)
    
    if max_tau is not None:
        max_shift = int(max_tau * fs)
    else:
        max_shift = len(sig1) // 2
    
    cc = np.concatenate([cc[-max_shift:], cc[:max_shift+1]])
    peak_idx = np.argmax(np.abs(cc))
    tau_samples = peak_idx - max_shift
    return tau_samples / fs

def estimate_doa(audio_chunk, fs):
    """Estimate Direction of Arrival."""
    n_channels = audio_chunk.shape[1]
    
    if n_channels >= 6:
        mics = audio_chunk[:, 1:5].astype(np.float64)
    elif n_channels >= 4:
        mics = audio_chunk[:, 0:4].astype(np.float64)
    else:
        return 0, 0
    
    max_tau = MIC_DISTANCE / SPEED_OF_SOUND * 2
    
    tau_02 = gcc_phat(mics[:, 0], mics[:, 2], fs, max_tau)
    tau_13 = gcc_phat(mics[:, 1], mics[:, 3], fs, max_tau)
    
    max_delay = MIC_DISTANCE / SPEED_OF_SOUND
    sin_theta1 = np.clip(tau_02 / max_delay, -1, 1)
    sin_theta2 = np.clip(tau_13 / max_delay, -1, 1)
    
    angle_rad = np.arctan2(sin_theta1, sin_theta2)
    angle_deg = np.degrees(angle_rad)
    if angle_deg < 0:
        angle_deg += 360
    
    energy = np.mean(mics**2)
    confidence = min(1.0, energy / 1000000)
    
    return angle_deg, confidence

def audio_callback(indata, frames, time_info, status):
    audio_queue.put(indata.copy())

def find_respeaker():
    """Find ReSpeaker audio device."""
    for i, dev in enumerate(sd.query_devices()):
        if dev['max_input_channels'] > 0:
            name = dev['name'].lower()
            if 'respeaker' in name or 'uac1.0' in name:
                return i, dev['max_input_channels'], dev['name']
    return None, None, None

class DoAVisualizer:
    def __init__(self, audio_stream):
        self.audio_stream = audio_stream
        self.angle_history = []
        self.history_size = 5
        self.current_angle = 0
        self.current_energy = 0
        
        # Setup plot
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(10, 10))
        self.fig.patch.set_facecolor('#0a0a0a')
        
        self.ax = self.fig.add_subplot(111, projection='polar')
        self.ax.set_facecolor('#0a0a0a')
        
        # Configure polar plot
        self.ax.set_theta_zero_location('N')  # 0° at top (front)
        self.ax.set_theta_direction(-1)  # Clockwise
        self.ax.set_ylim(0, 1.5)
        self.ax.set_yticks([])
        
        # Styling
        self.ax.grid(True, color='#333333', linestyle=':', linewidth=1, alpha=0.5)
        self.ax.set_xticks(np.radians([0, 45, 90, 135, 180, 225, 270, 315]))
        self.ax.set_xticklabels(['FRONT', 'FR', 'RIGHT', 'BR', 'BACK', 'BL', 'LEFT', 'FL'], 
                                color='#888888', fontsize=12, fontweight='bold')
        
        # Title
        self.title = self.ax.set_title("ReSpeaker DoA - Real-Time", 
                                       color='white', pad=20, fontsize=18, fontweight='bold')
        
        # Draw microphone positions
        mic_r = 0.3
        for i, angle in enumerate(MIC_ANGLES):
            x = mic_r * np.cos(angle)
            y = mic_r * np.sin(angle)
            # Convert to polar coordinates
            mic_theta = angle
            self.ax.plot([mic_theta], [mic_r], 'o', color='#00ff88', 
                        markersize=12, markeredgecolor='white', markeredgewidth=2,
                        zorder=10)
            # Label
            label_r = mic_r + 0.15
            self.ax.text(mic_theta, label_r, f'M{i}', 
                        ha='center', va='center', color='#00ff88', 
                        fontsize=10, fontweight='bold')
        
        # Center circle (microphone array outline)
        center_circle = Circle((0, 0), 0.35, transform=self.ax.transData._b,
                              fill=False, edgecolor='#00ff88', linewidth=2, alpha=0.3)
        self.ax.add_patch(center_circle)
        
        # Direction indicator (wedge)
        self.wedge = Wedge((0, 0), 1.3, 0, 0, width=0.3, 
                          transform=self.ax.transData._b,
                          facecolor='#00d2ff', edgecolor='#ffffff', 
                          linewidth=2, alpha=0.6, zorder=5)
        self.ax.add_patch(self.wedge)
        
        # Direction line
        self.direction_line, = self.ax.plot([], [], color='#00d2ff', 
                                            linewidth=4, zorder=8)
        
        # Info text
        self.info_text = self.ax.text(0.5, 1.15, '', transform=self.ax.transAxes,
                                      ha='center', va='top', color='#00d2ff',
                                      fontsize=24, fontweight='bold')
        
        # Energy bar background
        self.energy_text = self.ax.text(0.5, 0.05, '', transform=self.ax.transAxes,
                                        ha='center', va='bottom', color='#888888',
                                        fontsize=12)
    
    def update(self, frame):
        """Update visualization with new audio data."""
        if audio_queue.empty():
            return self.wedge, self.direction_line, self.info_text
        
        # Get audio
        data = audio_queue.get()
        
        # Calculate DoA
        angle, confidence = estimate_doa(data, RATE)
        
        # Smooth angle
        self.angle_history.append(angle)
        if len(self.angle_history) > self.history_size:
            self.angle_history.pop(0)
        
        # Circular averaging
        angles_rad = np.radians(self.angle_history)
        mean_sin = np.mean(np.sin(angles_rad))
        mean_cos = np.mean(np.cos(angles_rad))
        smooth_angle = np.degrees(np.arctan2(mean_sin, mean_cos))
        if smooth_angle < 0:
            smooth_angle += 360
        
        self.current_angle = smooth_angle
        
        # Energy
        self.current_energy = np.sqrt(np.mean(data.astype(float)**2))
        
        # Update wedge (beam)
        angle_rad = np.radians(smooth_angle)
        beam_width = 30  # degrees
        theta1 = smooth_angle - beam_width/2
        theta2 = smooth_angle + beam_width/2
        
        self.wedge.set_theta1(theta1)
        self.wedge.set_theta2(theta2)
        
        # Brightness based on energy
        alpha = min(0.8, self.current_energy / 10000)
        self.wedge.set_alpha(max(0.2, alpha))
        
        # Update direction line
        self.direction_line.set_data([angle_rad, angle_rad], [0, 1.3])
        
        # Update text
        self.info_text.set_text(f"{smooth_angle:.1f}°")
        
        # Energy indicator
        bar_len = min(int(self.current_energy / 300), 30)
        bar = "█" * bar_len
        self.energy_text.set_text(f"Energy: {bar}")
        
        return self.wedge, self.direction_line, self.info_text, self.energy_text

def main():
    print("=" * 60)
    print(" ReSpeaker DoA - Real-Time Visualization")
    print("=" * 60)
    
    # Find audio device
    print("\nSearching for ReSpeaker...")
    dev_idx, channels, dev_name = find_respeaker()
    
    if dev_idx is None:
        print("ERROR: ReSpeaker not found!")
        return
    
    print(f"  Device: {dev_name}")
    print(f"  Channels: {channels}")
    
    if channels < 4:
        print("ERROR: Need at least 4 channels!")
        return
    
    print("\nStarting visualization...")
    print("Make sounds from different directions around the mic array!")
    print("Close the window to stop.\n")
    
    # Start audio stream
    stream = sd.InputStream(device=dev_idx, channels=channels, samplerate=RATE,
                           blocksize=CHUNK_SIZE, callback=audio_callback, 
                           dtype='int16')
    
    with stream:
        # Create visualizer
        viz = DoAVisualizer(stream)
        
        # Animate
        ani = animation.FuncAnimation(viz.fig, viz.update, interval=50, 
                                     blit=False, cache_frame_data=False)
        
        try:
            plt.show()
        except KeyboardInterrupt:
            pass
    
    print("\nStopped.")

if __name__ == "__main__":
    main()
