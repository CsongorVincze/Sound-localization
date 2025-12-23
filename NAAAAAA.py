import numpy as np
import scipy
import time
import sounddevice as sd

DEVICE_STRING = "ReSpeaker 4 Mic Array (UAC1.0) , MME"
SAMPLE_RATE = 16000
BLOCK_SIZE = 1024  #ez nem fix h kell
NUM_CHANNELS = 1  # We want all 6 channels

fftdata = []

def audio_callback(indata, frames, time, status):
    if(status):
        print(status)
    
    fftvalue = np.abs(scipy.fft.fft(indata))
    fftdata.append(fftvalue)
    print(np.max(fftvalue))


try:
    with sd.InputStream(device=DEVICE_STRING, samplerate=SAMPLE_RATE, channels=NUM_CHANNELS, blocksize=0, callback=audio_callback):
        print("Listening... Press Ctrl+C to stop")
        while True:
            time.sleep(1)
except KeyboardInterrupt:
    print("\nInterrupted")


# -- EZ ITT CSAK PLOTOLAS EZT NEMTOM H JO-E --

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

if fftdata:
    fft_array = np.array(fftdata).squeeze()
    if fft_array.ndim == 1:
        fft_array = fft_array[np.newaxis, :]
    
    num_frames, n_fft = fft_array.shape
    freqs = np.fft.fftfreq(n_fft, 1/SAMPLE_RATE)[:n_fft // 2]

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)
    
    line, = ax.plot(freqs, fft_array[0, :n_fft // 2])
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude')
    ax.set_xlim(0, SAMPLE_RATE / 2)
    ax.grid(True)

    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
    slider = Slider(ax_slider, 'Frame', 0, num_frames - 1, valinit=0, valstep=1)

    def update(val):
        idx = int(slider.val)
        y_data = fft_array[idx, :n_fft // 2]
        line.set_ydata(y_data)
        ax.set_ylim(0, np.max(y_data) * 1.1 if np.max(y_data) > 0 else 1)
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()
