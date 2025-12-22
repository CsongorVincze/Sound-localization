import numpy as np
import sounddevice as sd
import time
import queue
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- Configuration ---
DEVICE_STRING = "ReSpeaker 4 Mic Array (UAC1.0) , MME"
SAMPLE_RATE = 16000
BLOCK_SIZE = 4096
NUM_CHANNELS = 6
MIC_CHANNELS = [0, 1, 2, 3]
SOUND_SPEED = 343.0
MIC_DIAMETER = 0.078 
MIC_RADIUS = MIC_DIAMETER / 2.0

# Beamforming Parameters
NUM_ANGLES = 72  # Resolution: 360 / 72 = 5 degrees
ANGLES = np.linspace(0, 2 * np.pi, NUM_ANGLES, endpoint=False)

# Frequency range for human speech
MIN_FREQ = 300
MAX_FREQ = 6000
VAD_THRESHOLD = 0.0001  # Lowered threshold for typical mic input levels

class BeamformingProcessor:
    def __init__(self, sample_rate, block_size):
        self.fs = sample_rate
        self.N = block_size
        self.window = np.hanning(self.N)
        
        # Microphone positions (assuming circular array)
        # Mic 0: (R, 0), Mic 1: (0, R), Mic 2: (-R, 0), Mic 3: (0, -R)
        self.mic_positions = np.array([
            [MIC_RADIUS, 0],      # Mic 0
            [0, MIC_RADIUS],      # Mic 1
            [-MIC_RADIUS, 0],     # Mic 2
            [0, -MIC_RADIUS]      # Mic 3
        ])
        
        self.steering_vectors, self.freq_bins = self._precompute_steering_vectors()

    def _precompute_steering_vectors(self):
        """
        Precomputes steering vectors for all defined angles and frequencies.
        Returns:
            numpy.ndarray: Shape (num_freqs, num_angles, num_mics)
            numpy.ndarray: Indices of the used frequency bins
        """
        # Frequency bins
        freqs = np.fft.rfftfreq(self.N, d=1/self.fs)
        
        # Frequency Masking: Select only relevant frequencies
        freq_mask = (freqs >= MIN_FREQ) & (freqs <= MAX_FREQ)
        target_freqs = freqs[freq_mask]
        target_indices = np.where(freq_mask)[0]
        
        # Wave numbers: k = 2*pi*f / c
        k = 2 * np.pi * target_freqs / SOUND_SPEED  # Shape: (num_target_freqs,)
        
        # Unit vectors for each look direction
        # Shape: (num_angles, 2)
        u = np.array([np.cos(ANGLES), np.sin(ANGLES)]).T
        
        # Projections
        projections = np.dot(self.mic_positions, u.T) # Shape: (num_mics, num_angles)
        
        # Phase shifts
        # Shape: (num_target_freqs, num_mics, num_angles)
        phases = -np.einsum('f,ma->fma', k, projections)
        
        # Steering vectors: exp(j * phi)
        steering_vectors = np.exp(1j * phases)
        
        # Transpose to (num_target_freqs, num_angles, num_mics)
        return steering_vectors.transpose(0, 2, 1), target_indices

    def process(self, indata):
        """
        Process a block of audio data and return the estimated angle.
        """
        mics = indata[:, MIC_CHANNELS]
        
        # VAD: Check energy
        energy = np.mean(mics**2)
        if energy < VAD_THRESHOLD:
            return None

        mics_w = mics * self.window[:, np.newaxis]
        
        # Compute STFT
        # Shape: (num_freqs, num_mics)
        full_spectra = np.fft.rfft(mics_w, axis=0)
        
        # Select only relevant frequencies
        spectra = full_spectra[self.freq_bins, :]
        
        # PHAT Weighting (Whitening)
        # Normalize magnitude to 1 (preserve phase)
        spectra = spectra / (np.abs(spectra) + 1e-10)
        
        # SRP-PHAT (Steered Response Power with Phase Transform)
        # Output(theta) = Sum_f Sum_m ( Spectra(f, m) * Steering(f, theta, m) )
        
        # Broadcast spectra to (F, 1, M)
        # Multiply by steering vectors (F, A, M)
        # Sum over mics (axis 2)
        beamformed_spectra = np.sum(spectra[:, np.newaxis, :] * self.steering_vectors, axis=2)
        
        # Compute Power
        # Sum of magnitude squared over frequencies
        # Shape: (A,)
        power_spectrum = np.sum(np.abs(beamformed_spectra)**2, axis=0)
        
        # Find angle with maximum power
        max_idx = np.argmax(power_spectrum)
        estimated_angle = ANGLES[max_idx]
        
        return estimated_angle

def main():
    processor = BeamformingProcessor(SAMPLE_RATE, BLOCK_SIZE)
    
    # List to store recorded data: (time, angle)
    recorded_data = []
    start_time = time.time()
    
    print(f"Listening on {DEVICE_STRING}...")
    print("Using SRP-PHAT Beamforming")
    print("Recording... Press Ctrl+C to stop and view the animation.")

    def callback(indata, frames, time_info, status):
        if status:
            print(status)
        
        angle = processor.process(indata)
        
        if angle is not None:
            # Store timestamp and angle
            # Adjust angle for plotting
            plot_angle = np.pi/2 - angle
            plot_angle = (plot_angle + np.pi) % (2 * np.pi) - np.pi
            
            elapsed = time.time() - start_time
            recorded_data.append((elapsed, plot_angle))
        else:
            # Store None to indicate silence/no detection
            elapsed = time.time() - start_time
            recorded_data.append((elapsed, None))

    # --- Recording Phase ---
    try:
        with sd.InputStream(
            device=DEVICE_STRING,
            channels=NUM_CHANNELS,
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            callback=callback
        ):
            while True:
                sd.sleep(100)
                
    except KeyboardInterrupt:
        print("\nRecording stopped.")
        valid_frames = sum(1 for _, a in recorded_data if a is not None)
        print(f"Captured {len(recorded_data)} frames ({valid_frames} valid).")
    except Exception as e:
        print(f"\nError: {e}")
        return

    if not recorded_data:
        print("No data recorded.")
        return

    # --- Playback/Animation Phase ---
    print("Starting animation...")
    
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_title("Recorded Beamforming DoA")
    
    # The needle
    line, = ax.plot([], [], lw=3, color='blue')
    time_text = ax.text(0, 0, '', transform=ax.transAxes)
    
    # Smoothing state
    current_angle = [0.0]
    alpha = 0.2
    
    def update(frame_idx):
        if frame_idx >= len(recorded_data):
            return line, time_text
            
        t, target_angle = recorded_data[frame_idx]
        
        if target_angle is not None:
            # Smoothing
            diff = target_angle - current_angle[0]
            if diff > np.pi:
                diff -= 2*np.pi
            elif diff < -np.pi:
                diff += 2*np.pi
            
            current_angle[0] += diff * alpha
            current_angle[0] = (current_angle[0] + np.pi) % (2 * np.pi) - np.pi
            
            line.set_data([current_angle[0], current_angle[0]], [0, 1])
            line.set_color('blue')
        else:
            # Optional: Hide line or change color when silence
            # line.set_color('gray') # or keep last position
            pass
            
        ax.set_title(f"Time: {t:.2f} s")
        return line, time_text

    # Calculate interval based on block size and sample rate to match real-time speed roughly
    # Block duration = N / fs
    block_duration_ms = (BLOCK_SIZE / SAMPLE_RATE) * 1000
    
    ani = animation.FuncAnimation(
        fig, 
        update, 
        frames=len(recorded_data), 
        interval=block_duration_ms, 
        blit=False, # blit=False often safer for text updates
        repeat=False
    )
    
    plt.show()

if __name__ == "__main__":
    main()
