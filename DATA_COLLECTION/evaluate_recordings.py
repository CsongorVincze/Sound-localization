import pandas as pd
import numpy as np
import soundfile as sf
import pyroomacoustics as pra
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import itertools
import sys

# --- CONFIGURATION ---
SESSION_PATH = Path("DATA_COLLECTION/sessions/session_20260521_005414")
FS = 16000
MIC_RADIUS = 0.035 # 3.5cm
# ReSpeaker 4-mic array angles: 0, 90, 180, 270 degrees
MIC_POS = np.array([
    [MIC_RADIUS * np.cos(np.deg2rad(a)), MIC_RADIUS * np.sin(np.deg2rad(a)), 0] 
    for a in [0, 90, 180, 270]
]).T

# Scenario A: 2 pairs (orthogonal cross)
PAIRS_2 = [(0, 2), (1, 3)]
# Scenario B: 4 pairs (square perimeter)
PAIRS_4 = [(0, 1), (1, 2), (2, 3), (3, 0)]
# Scenario C: 6 pairs (all combinations)
PAIRS_6 = list(itertools.combinations(range(4), 2))

def circular_dist(a, b):
    d = np.abs(a - b) % 360
    return np.where(d > 180, 360 - d, d)

def run_srp_phat(audio, pairs, nfft=512):
    """
    Robust SRP-PHAT implementation for ReSpeaker data.
    Input 'audio' shape: (samples, channels)
    """
    # 1. STFT: pra expects (samples, channels)
    # Returns X with shape (frames, bins, channels)
    X = pra.transform.stft.analysis(audio, L=nfft, hop=nfft//2)
    
    # Check if we got the expected 3D array (frames, bins, channels)
    if X.ndim < 3:
        # If it returned (frames, bins), we only have one channel
        return 0.0

    # Reshape to (channels, bins, frames) for processing
    X = np.swapaxes(X, 0, 2)
    
    # 2. Grid search
    angles = np.linspace(0, 2*np.pi, 360, endpoint=False)
    grid_points = np.array([np.cos(angles), np.sin(angles), np.zeros_like(angles)])
    
    total_energy = np.zeros(360)
    
    # 3. SRP Logic per pair
    # Speed of sound
    c = 343.0
    freqs = np.fft.rfft(np.zeros(nfft)).size
    f_vec = np.linspace(0, FS/2, freqs).reshape(-1, 1)

    for i, j in pairs:
        # Cross-spectrum: W * X1 * conj(X2)
        # GCC-PHAT weight: 1 / |X1 * conj(X2)|
        cross_power = X[i] * np.conj(X[j])
        gcc_phat = cross_power / (np.abs(cross_power) + 1e-9)
        
        # Time-average of weighted cross-spectrum
        X_avg = np.mean(gcc_phat, axis=1) # (bins)
        
        # TDOA for this pair at every grid angle
        # delta_d = dist(mic_i) - dist(mic_j) = dot(p_i - p_j, u_theta)
        dist_diffs = np.dot(MIC_POS[:, i] - MIC_POS[:, j], grid_points)
        tau = dist_diffs / c
        
        # Steering vector phase shifts
        steering_phases = np.exp(1j * 2 * np.pi * f_vec * tau)
        
        # Contribution to SRP: sum over frequencies
        pair_energy = np.real(np.dot(X_avg, steering_phases))
        total_energy += pair_energy

    return np.degrees(angles[np.argmax(total_energy)]) % 360

def evaluate():
    meta_path = SESSION_PATH / "metadata.csv"
    if not meta_path.exists():
        print(f"[!] Error: {meta_path} not found.")
        return

    df = pd.read_csv(meta_path)
    df = df[df['clip_type'] == 'source'].copy()
    
    if df.empty:
        print("[!] No 'source' recordings found in metadata.")
        return

    scenarios = {
        'SRP (2 pairs)': PAIRS_2,
        'SRP (4 pairs)': PAIRS_4,
        'SRP (6 pairs)': PAIRS_6
    }
    
    data_results = []
    
    print(f"[*] Analyzing {len(df)} files from {SESSION_PATH.name}...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        wav_p = SESSION_PATH / row['recording']
        if not wav_p.exists(): continue
        
        try:
            audio, fs = sf.read(wav_p)
            
            # Ensure 4-channel input (our raw mic count)
            if audio.ndim == 1:
                print(f"Skipping mono file: {wav_p}")
                continue
            if audio.shape[1] > 4:
                audio = audio[:, :4]
            elif audio.shape[1] < 4:
                print(f"Skipping file with {audio.shape[1]} channels: {wav_p}")
                continue
                
            entry = {
                'sound_type': row.get('sound_type', 'unknown'),
                'doa_degrees': row['doa_degrees']
            }
            
            for name, pairs in scenarios.items():
                pred = run_srp_phat(audio, pairs)
                err = circular_dist(pred, row['doa_degrees'])
                entry[f'{name}_pred'] = pred
                entry[f'{name}_err'] = err
                
            data_results.append(entry)
        except Exception as e:
            print(f"Error processing {wav_p}: {e}")
            continue

    if not data_results:
        print("[!] No valid recordings were processed.")
        return

    res_df = pd.DataFrame(data_results)

    # --- Analysis Summary ---
    sound_types = res_df['sound_type'].unique()
    print("\n" + "="*50)
    print(" PERFORMANCE SUMMARY (MSE)")
    print("="*50)
    summary = []
    for st in sound_types:
        st_df = res_df[res_df['sound_type'] == st]
        row_summary = {'Sound Type': st}
        for name in scenarios:
            errs = st_df[f'{name}_err']
            mse = np.mean(errs**2)
            mae = np.mean(errs)
            row_summary[f'{name} MSE'] = round(mse, 2)
            row_summary[f'{name} MAE'] = round(mae, 1)
        summary.append(row_summary)
    
    print(pd.DataFrame(summary).to_string(index=False))

    # --- Plots ---
    plt.figure(figsize=(12, 6))
    x = np.arange(len(sound_types))
    width = 0.25
    for i, name in enumerate(scenarios):
        mses = [np.mean(res_df[res_df['sound_type'] == st][f'{name}_err']**2) for st in sound_types]
        plt.bar(x + i*width, mses, width, label=name)
    plt.xlabel('Sound Type'); plt.ylabel('MSE'); plt.title('DoA Accuracy by Sound Category')
    plt.xticks(x + width, sound_types); plt.legend(); plt.grid(axis='y', alpha=0.3)
    plt.savefig('evaluation_mse_overall.png')

    plt.figure(figsize=(12, 6))
    angles = sorted(res_df['doa_degrees'].unique())
    for name in scenarios:
        err_per_angle = [res_df[res_df['doa_degrees'] == a][f'{name}_err'].mean() for a in angles]
        plt.plot(angles, err_per_angle, marker='o', label=name, markersize=3)
    plt.xlabel('Ground Truth Angle (°)'); plt.ylabel('Avg Error (°)'); plt.title('Error Distribution per Angle')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig('evaluation_error_per_angle.png')

    print(f"\n[*] Evaluation complete. Plots saved as 'evaluation_mse_overall.png' and 'evaluation_error_per_angle.png'.")

if __name__ == "__main__":
    evaluate()
