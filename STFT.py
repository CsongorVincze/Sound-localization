import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

# Read the WAV file
sample_rate, audio_data = wavfile.read("lalalalala_1.wav")
# sample_rate -> freki
# audio_file -> legnyomas


audio_num = np.array(audio_data[180224:184832])  # hanganyag np formatumban

print(audio_num.shape)
print(sample_rate)

firgiforgi = np.zeros_like(audio_num, dtype=complex)

k = np.arange(50, 5000, 1)  # ez ilyen frekis allito a DFT-hez
print(k)
summasummarom = np.zeros_like(
    k, dtype=complex
)  # tarolja a kulonbozo frekikhez a DFT erteket

for na in range(k.size):
    print(
        f"\rProcessing frequency {k[na]} of {k[-1]} ({(na + 1) / len(k) * 100:.1f}%)",
        end="",
    )
    for i in range(audio_num.size):
        firgiforgi[i] = np.exp(-1.0j * 2 * np.pi * k[na] * i / audio_num.size)
        summasummarom[na] += firgiforgi[i] * audio_num[i]


plt.figure()
plt.plot(k, np.abs(summasummarom))
plt.xlabel("Frequency (k)")
plt.ylabel("Magnitude")
plt.title("DFT Magnitude vs Frequency")
plt.grid(True)
plt.show()
