import numpy as np
import scipy
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

    # atmegyunk minden mikrofonnal freq domainbe
    freqs_ = np.vstack([scipy.fft.fft(indata[:, i]) for i in range(1, 5)])
    freqs_conj = np.conj(freqs_)

    S = np.einsum("ij, kj -> ikj", freqs_, freqs_conj)

    epsilon = 0.001
    R = scipy.fft.ifft(S / (np.abs(S) + epsilon)) #!itt lehet hogy rossz indexre ifft-zunk
    #! meg itt nem vesztunk el minden adatot azzal h leosztunk mert az exp(i*omega*t) az mindig megvan nem?
    print(np.argmax(R) / SAMPLE_RATE)

    # print(np.max(np.abs(freqs_), axis=1))  # kiirjuk a max ertekeket

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
