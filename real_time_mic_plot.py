import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import queue
import sys

DEVICE_STRING = "ReSpeaker 4 Mic Array (UAC1.0) , MME"
SAMPLE_RATE = 16000
BLOCK_SIZE = 1024
NUM_CHANNELS = 4
WINDOW_SIZE = 8000  # Show last half second (assuming 16k sample rate)

# Queue to transport data from the callback to the plotting thread
q = queue.Queue()

def audio_callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    # indata is (frames, channels). Put a copy in the queue.
    q.put(indata.copy())

def update_plot(frame):
    """This is called by matplotlib for each plot update."""
    global plotdata
    while True:
        try:
            data = q.get_nowait()
        except queue.Empty:
            break
        shift = len(data)
        plotdata = np.roll(plotdata, -shift, axis=0)
        plotdata[-shift:, :] = data

    for column, line in enumerate(lines):
        line.set_ydata(plotdata[:, column])
    return lines

# Initialize plot data buffer
plotdata = np.zeros((WINDOW_SIZE, NUM_CHANNELS))

fig, axes = plt.subplots(NUM_CHANNELS, 1, sharex=True, figsize=(10, 8))
if NUM_CHANNELS == 1:
    axes = [axes] # Ensure iterable

lines = []
for i, ax in enumerate(axes):
    line, = ax.plot(plotdata[:, i])
    lines.append(line)
    ax.set_ylim(-0.5, 0.5) # Initial limits, sounddevice returns float32 in [-1, 1] usually
    ax.set_ylabel(f'Ch {i+1}')
    ax.grid(True)

axes[-1].set_xlabel('Samples')
fig.suptitle('Real-time Raw Mic Data')
fig.tight_layout()

print(f"Starting stream on {DEVICE_STRING} with {NUM_CHANNELS} channels...")

try:
    with sd.InputStream(device=DEVICE_STRING, channels=NUM_CHANNELS, 
                        samplerate=SAMPLE_RATE, callback=audio_callback, blocksize=BLOCK_SIZE):
        ani = FuncAnimation(fig, update_plot, interval=30, blit=True, cache_frame_data=False)
        plt.show()
except Exception as e:
    print(f"\nError: {e}")
    print("NB: Ensure the device string is correct and the device supports the requested number of channels.")
