import pandas as pd
import numpy as np
import pyroomacoustics as pra
from scipy.io import wavfile
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

# Configuration
SESSION_DIR = "DATA_COLLECTION/sessions/session_20260517_161547"
METADATA_FILE = os.path.join(SESSION_DIR, "metadata.csv")
FS = 16000
NFFT = 512
FREQS = [300, 3000]

# ReSpeaker v2.0 geometry
MIC_SPACING = 0.0465
MIC_RADIUS = MIC_SPACING / np.sqrt(2)
MIC_ANGLE_DEG = np.array([135, 225, 315, 45]) # M0, M1, M2, M3

# mic_pos for PRA (x=Right, y=Front)
mic_pos_4 = np.array([
    [MIC_RADIUS * np.sin(np.radians(ang)), MIC_RADIUS * np.cos(np.radians(ang))]
    for ang in MIC_ANGLE_DEG
]).T # (2, 4)

# 2-mic version: using only M0 and M2 (diagonal pair)
mic_pos_2 = mic_pos_4[:, [0, 2]] # (2, 2)

def angular_error(a_rad, b_rad):
    err = a_rad - b_rad
    return (err + np.pi) % (2 * np.pi) - np.pi

def evaluate():
    print(f"Loading metadata from {METADATA_FILE}...")
    df = pd.read_csv(METADATA_FILE)
    df = df[df['clip_type'] == 'source']
    df = df[(df['sweep'] > 2) & (df['sweep'] < 20)]
    
    print(f"Found {len(df)} recordings to process.")
    
    algorithms = ["4-Mic SRP", "2-Mic SRP (M0,M2)"]
    raw_results = {alg: [] for alg in algorithms}
    degree_results = {}
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        wav_path = os.path.join(SESSION_DIR, row['recording'])
        if not os.path.exists(wav_path): continue
        fs, audio = wavfile.read(wav_path)
        audio = audio.astype(np.float32) / 32768.0
        if audio.shape[1] > 4: audio = audio[:, :4]
        
        # 4-mic STFT
        X4 = np.array([pra.transform.stft.analysis(audio[:, i], L=NFFT, hop=NFFT//2) for i in range(4)])
        X4 = np.swapaxes(X4, 1, 2)
        
        # 2-mic STFT (Channels 0 and 2)
        X2 = X4[[0, 2], :, :]
        
        true_doa_respeaker = row['doa_degrees']
        true_doa_pra = np.radians((90 - true_doa_respeaker) % 360)
        
        # 1. 4-Mic SRP
        try:
            doa4 = pra.doa.SRP(mic_pos_4, fs, nfft=NFFT, num_src=1)
            doa4.locate_sources(X4, freq_range=FREQS)
            if len(doa4.azimuth_recon) > 0:
                err = np.degrees(np.abs(angular_error(doa4.azimuth_recon[0], true_doa_pra)))
                raw_results["4-Mic SRP"].append(err**2)
                if true_doa_respeaker not in degree_results: degree_results[true_doa_respeaker] = {alg: [] for alg in algorithms}
                degree_results[true_doa_respeaker]["4-Mic SRP"].append(err**2)
        except: pass

        # 2. 2-Mic SRP
        try:
            doa2 = pra.doa.SRP(mic_pos_2, fs, nfft=NFFT, num_src=1)
            doa2.locate_sources(X2, freq_range=FREQS)
            if len(doa2.azimuth_recon) > 0:
                err = np.degrees(np.abs(angular_error(doa2.azimuth_recon[0], true_doa_pra)))
                raw_results["2-Mic SRP (M0,M2)"].append(err**2)
                if true_doa_respeaker not in degree_results: degree_results[true_doa_respeaker] = {alg: [] for alg in algorithms}
                degree_results[true_doa_respeaker]["2-Mic SRP (M0,M2)"].append(err**2)
        except: pass

    print("\n--- Comparison Results ---")
    for alg in algorithms:
        mse = np.mean(raw_results[alg])
        print(f"{alg}: RMSE = {np.sqrt(mse):.2f} degrees")

    # Plotting
    sorted_degrees = sorted(degree_results.keys())
    plt.figure(figsize=(12, 6))
    for alg in algorithms:
        mse_deg = [np.mean(degree_results[d][alg]) for d in sorted_degrees]
        plt.plot(sorted_degrees, mse_deg, label=alg)
    
    plt.xlabel("Ground Truth DoA (degrees)")
    plt.ylabel("MSE (degrees^2)")
    plt.title("DoA Accuracy: 4 Microphones vs 2 Microphones")
    plt.legend()
    plt.grid(True)
    plt.savefig("doa_comparison_2mic_4mic.png")
    print("Saved doa_comparison_2mic_4mic.png")
    plt.show()

if __name__ == "__main__":
    evaluate()
