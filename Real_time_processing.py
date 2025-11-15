import numpy as np
import time
import sounddevice as sd

# --- Parameters ---
DEVICE_STRING = "ReSpeaker 4 Mic Array (UAC1.0) , MME"
SAMPLE_RATE = 16000
BLOCK_SIZE = 1024  # <-- TUNE THIS FOR LATENCY. Try 256, 512, 1024.
NUM_CHANNELS = 6  # We want all 6 channels
# RAW_MIC_CHANNEL = 1  # We'll process the raw audio from mic 1


# This is your real-time processing function!
def audio_callback(indata, frames, time, status):
    """
    This function is called by the audio thread for each new block of data.
    'indata' is a NumPy array of the audio data.
    Your processing MUST be fast enough to finish before the next block arrives.
    """
    if status:
        print(status)  # Print any errors (e.g., buffer underruns)

    # --- YOUR REAL-TIME ANALYSIS GOES HERE ---

    # stack channels and compute FFTs for all at once
    mic_stack = np.vstack(
        [
            indata[:, 1],
            indata[:, 2],
            indata[:, 3],
            indata[:, 4],
        ]  #! itt nemtom h jok-e az indexek
    )  # shape (4, frames)
    fft_data = np.abs(np.fft.rfft(mic_stack, axis=1))  # shape (4, n_fft_bins)

    # compute frequency axis and dominant frequency per channel (as ints)
    freqs = np.fft.rfftfreq(indata[:, 1].size, d=1.0 / SAMPLE_RATE)
    peak_idxs = np.argmax(fft_data, axis=1)  # one index per channel
    dominant_freq = freqs[peak_idxs].astype(int)  # array of ints, shape (4,)

    print(dominant_freq)

    # --- END OF ANALYSIS ---


# --- Main Program ---
try:
    # Open the audio stream
    with sd.InputStream(
        device=DEVICE_STRING,
        channels=NUM_CHANNELS,
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        callback=audio_callback,  # This is the magic!
    ):
        print(f"Starting real-time analysis on '{DEVICE_STRING}'...")
        print("Press Ctrl+C to stop.")
        while True:
            # The main thread can sleep or do other work.
            # The audio_callback is running in the background.
            time.sleep(1)

except KeyboardInterrupt:
    print("\nStopping...")
except Exception as e:
    print(f"An error occurred: {e}")
