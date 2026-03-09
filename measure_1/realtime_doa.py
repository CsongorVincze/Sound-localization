import queue
import sys
import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from my_algos import get_gcc_phat_angle, get_srp_phat_angle, get_basic_cc_angle

# --- Configuration ---
SAMPLE_RATE = 16000
CHUNK_DURATION = 0.5  # Compute DoA every 0.5 seconds of audio for stable output
BLOCK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)

# Name, function, color, and line length for the polar plot
ALGORITHMS = [
    ("GCC-PHAT", get_gcc_phat_angle, "#00d2ff", 1.0),
    ("SRP-PHAT", get_srp_phat_angle, "#4ecdc4", 0.8),
    ("Basic CC", get_basic_cc_angle, "#ff6b6b", 0.6),
]

# Audio streaming queue
q = queue.Queue()

def audio_callback(indata, frames, time, status):
    """This is called continuously for each audio block from the microphone."""
    if status:
        print(status, file=sys.stderr)
    # Put a copy of the data into the queue
    q.put(indata.copy())

def main():
    # --- Device Discovery ---
    respeaker_id = None
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if dev['max_input_channels'] >= 4:
            name = dev['name'].lower()
            if 'respeaker' in name or 'uac1.0' in name or 'seeed' in name:
                respeaker_id = i
                print(f"Found ReSpeaker: {dev['name']} (ID: {i})")
                break

    if respeaker_id is None:
        print("WARNING: ReSpeaker not found! Attempting to use default input device.")
        respeaker_id = sd.default.device[0]

    device_info = sd.query_devices(respeaker_id, 'input')
    channels = device_info['max_input_channels']
    print(f"Using device ID {respeaker_id} with {channels} channels at {SAMPLE_RATE}Hz")

    # --- Plot Setup ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_zero_location('N') # 0 degrees at top (Front)
    ax.set_theta_direction(-1)      # Clockwise positive

    lines = {}
    for name, _, color, length in ALGORITHMS:
        # We use a line extending from center to 'length'
        lines[name], = ax.plot([], [], color=color, linewidth=4, label=name)
        lines[name].length = length

    ax.set_ylim(0, 1.1)
    ax.set_yticks([]) # Hide radial ticks
    ax.set_xticks(np.radians(np.arange(0, 360, 45)))
    ax.set_xticklabels(['Front (0°)', '45°', 'Right (90°)', '135°', 'Back (180°)', '225°', 'Left (270°)', '315°'])

    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1))
    plt.title("Real-Time Direction of Arrival (DoA)", va='bottom', pad=20, fontsize=14, fontweight='bold')

    # State for audio accumulation
    audio_buffer = np.zeros((0, channels))

    def update_plot(frame):
        nonlocal audio_buffer
        
        # Empty the queue into our buffer
        while True:
            try:
                data = q.get_nowait()
                audio_buffer = np.vstack((audio_buffer, data))
            except queue.Empty:
                break
                
        # If we have enough data to process a chunk
        if len(audio_buffer) >= BLOCK_SIZE:
            # Take the most recent BLOCK_SIZE samples
            process_data = audio_buffer[-BLOCK_SIZE:]
            
            # Clear buffer to keep it truly real-time and prevent lag accumulation
            audio_buffer = np.zeros((0, channels)) 
            
            # ReSpeaker mics are generally channels 1 to 4 (idx 1:5) when 6 channels are exposed
            if channels >= 6:
                raw_audio = process_data[:, 1:5].astype(np.float64)
            elif channels >= 4:
                raw_audio = process_data[:, 0:4].astype(np.float64)
            else:
                print(f"Error: Not enough channels ({channels}) for 4-mic DoA!")
                return lines.values()
            
            angles_str = []
            for name, algo_func, color, length in ALGORITHMS:
                try:
                    angle_deg = algo_func(raw_audio, SAMPLE_RATE)
                    angle_rad = np.radians(angle_deg)
                    
                    # Update line data to point to the new angle
                    lines[name].set_data([angle_rad, angle_rad], [0, length])
                    angles_str.append(f"{name}: {angle_deg:5.1f}°")
                except Exception as e:
                    pass
                    
            # Output to console cleanly
            if angles_str:
                sys.stdout.write("\r" + " | ".join(angles_str).ljust(60))
                sys.stdout.flush()
                
        return lines.values()

    # --- Start Streaming & Animation ---
    try:
        stream = sd.InputStream(
            device=respeaker_id, 
            channels=channels,
            samplerate=SAMPLE_RATE, 
            callback=audio_callback,
            blocksize=int(SAMPLE_RATE * 0.1) # Read in 100ms chunks internally
        )
        with stream:
            print("\nStarting real-time plot. Close the window to stop.")
            ani = FuncAnimation(fig, update_plot, interval=50, blit=False, cache_frame_data=False)
            plt.show()
    except Exception as e:
        print(f"\nFailed to start audio stream: {e}")

if __name__ == "__main__":
    main()
