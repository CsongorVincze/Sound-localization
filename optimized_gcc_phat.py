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

class GCCPHATProcessor:
    def __init__(self, sample_rate, block_size):
        self.fs = sample_rate
        self.N = block_size
        self.window = np.hanning(self.N)
        self.epsilon = 1e-6

    def process(self, indata):
        mics = indata[:, MIC_CHANNELS]
        mics_w = mics * self.window[:, np.newaxis]
        spectra = np.fft.rfft(mics_w, axis=0)
        
        tau_02 = self.compute_gcc_phat(spectra[:, 0], spectra[:, 2])
        tau_13 = self.compute_gcc_phat(spectra[:, 1], spectra[:, 3])
        
        return tau_02, tau_13

    def compute_gcc_phat(self, sig1_fft, sig2_fft):
        R_12 = sig1_fft * np.conj(sig2_fft)
        cc = R_12 / (np.abs(R_12) + self.epsilon)
        cc_time = np.fft.irfft(cc)
        shift = np.argmax(cc_time)
        if shift > self.N // 2:
            shift -= self.N
        tau = shift / self.fs
        return tau

def main():
    processor = GCCPHATProcessor(SAMPLE_RATE, BLOCK_SIZE)
    q = queue.Queue()
    
    print(f"Listening on {DEVICE_STRING}...")
    print("Press Ctrl+C to stop.")

    def callback(indata, frames, time_info, status):
        if status:
            print(status)
        tau_x, tau_y = processor.process(indata)
        
        # Calculate angle
        # tau_x > 0 -> Source in front (0 deg)
        # tau_y > 0 -> Source to right (90 deg)
        angle = np.arctan2(tau_y, tau_x)
        
        try:
            q.put_nowait(angle)
        except queue.Full:
            pass

    # --- Visualization Setup ---
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.set_theta_zero_location('N') # 0 degrees at North
    ax.set_theta_direction(-1)      # Clockwise
    ax.set_ylim(0, 1)
    ax.set_yticks([]) # Hide radial ticks
    ax.set_title("Sound Source Localization")
    
    # The needle
    line, = ax.plot([], [], lw=3, color='red')
    
    # Smoothing factor (simple exponential moving average)
    current_angle = [0.0] 
    alpha = 0.2 # Smoothing coefficient (0.0 to 1.0)

    def update(frame):
        try:
            # Get the latest angle from the queue
            # We drain the queue to get the most recent value
            target_angle = None
            while not q.empty():
                target_angle = q.get_nowait()
            
            if target_angle is not None:
                # Smooth the angle
                # Handle wrapping around PI/-PI is tricky with simple average, 
                # but for now let's just smooth the vector or the angle directly if close.
                # Simple approach: just update
                current_angle[0] = current_angle[0] * (1 - alpha) + target_angle * alpha
                
                # Update plot
                line.set_data([current_angle[0], current_angle[0]], [0, 1])
        except:
            pass
        return line,

    ani = animation.FuncAnimation(fig, update, interval=30, blit=True)

    try:
        with sd.InputStream(
            device=DEVICE_STRING,
            channels=NUM_CHANNELS,
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            callback=callback
        ):
            plt.show()
                
    except KeyboardInterrupt:
        print("\nStopped.")
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    main()
