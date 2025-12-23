import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.fft
import queue
import sys

DEVICE_STRING = "ReSpeaker 4 Mic Array (UAC1.0) , MME"
SAMPLE_RATE = 16000
BLOCK_SIZE = 1024
NUM_CHANNELS = 4

# Queue for thread-safe data transfer
q = queue.Queue()

def audio_callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    q.put(indata.copy())

def update_plot(frame):
    current_data = None
    while True:
        try:
            current_data = q.get_nowait()
        except queue.Empty:
            break
            
    if current_data is None:
        return lines
        
    # Perform FFT on the latest block
    # data shape: (BLOCK_SIZE, NUM_CHANNELS)
    N = len(current_data)
    
    # Calculate FFT along the time axis (axis 0)
    yf = scipy.fft.fft(current_data, axis=0)
    xf = scipy.fft.fftfreq(N, 1/SAMPLE_RATE)[:N//2]
    
    # Magnitude
    mag = np.abs(yf[:N//2, :]) # Shape (N/2, NUM_CHANNELS)
    
    # Update lines
    for i, line in enumerate(lines):
        line.set_data(xf, mag[:, i])
        
    return lines

fig, axes = plt.subplots(NUM_CHANNELS, 1, figsize=(10, 10), sharex=True)
if NUM_CHANNELS == 1:
    axes = [axes]

lines = []
# Pre-calculate x frequency axis for initial setup (assuming constant block size)
xf_init = scipy.fft.fftfreq(BLOCK_SIZE, 1/SAMPLE_RATE)[:BLOCK_SIZE//2]

for i, ax in enumerate(axes):
    line, = ax.plot([], [], lw=1)
    lines.append(line)
    ax.set_xlim(0, SAMPLE_RATE / 2)
    
    # Y-limit: FFT magnitude depends on signal strength. 
    # Arbitrary starting value, user might need to adjust based on volume.
    ax.set_ylim(0, 1000) 
    
    ax.set_ylabel(f'Ch {i+1}')
    ax.grid(True)

axes[-1].set_xlabel('Frequency (Hz)')
fig.suptitle(f'Real-Time FFT ({NUM_CHANNELS} Channels)')

print(f"Starting FFT stream on {DEVICE_STRING} with {NUM_CHANNELS} channels...")

try:
    with sd.InputStream(device=DEVICE_STRING, channels=NUM_CHANNELS, 
                        samplerate=SAMPLE_RATE, callback=audio_callback, blocksize=BLOCK_SIZE):
        ani = FuncAnimation(fig, update_plot, interval=30, blit=True, cache_frame_data=False)
        plt.show()
except Exception as e:
    print(f"\nError: {e}")
    print("Ensure the device supports the requested number of channels.")
