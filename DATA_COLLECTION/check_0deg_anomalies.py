import pandas as pd
import numpy as np
import pyroomacoustics as pra
from scipy.io import wavfile
import os
from tqdm import tqdm

# Configuration
SESSION_DIR = "DATA_COLLECTION/sessions/session_20260517_161547"
METADATA_FILE = os.path.join(SESSION_DIR, "metadata.csv")
FS = 16000
NFFT = 512
FREQS = [300, 3000]

# Geometry (working one)
MIC_SPACING = 0.0465
MIC_RADIUS = MIC_SPACING / np.sqrt(2)
MIC_ANGLE_DEG = np.array([135, 225, 315, 45])
mic_pos = np.array([
    [MIC_RADIUS * np.sin(np.radians(ang)), MIC_RADIUS * np.cos(np.radians(ang))]
    for ang in MIC_ANGLE_DEG
]).T

def angular_error(a_rad, b_rad):
    err = a_rad - b_rad
    return (err + np.pi) % (2 * np.pi) - np.pi

def analyze_0deg():
    df = pd.read_csv(METADATA_FILE)
    df_0 = df[(df['clip_type'] == 'source') & (df['doa_degrees'] == 0.0) & (df['sweep'] > 1) & (df['sweep'] < 20)]
    
    print(f"Analyzing {len(df_0)} recordings at 0 degrees...")
    
    results = []
    for idx, row in tqdm(df_0.iterrows(), total=len(df_0)):
        wav_path = os.path.join(SESSION_DIR, row['recording'])
        if not os.path.exists(wav_path): continue
        fs, audio = wavfile.read(wav_path)
        audio = audio.astype(np.float32) / 32768.0
        
        is_clipped = np.any(np.abs(audio) >= 0.99)
        rms = np.sqrt(np.mean(audio**2, axis=0))
        
        X = np.array([pra.transform.stft.analysis(audio[:, i], L=NFFT, hop=NFFT//2) for i in range(4)])
        X = np.swapaxes(X, 1, 2)
        true_doa_pra = np.radians(90.0)
        
        # SRP
        err_srp = np.nan
        try:
            doa_srp = pra.doa.SRP(mic_pos, fs, nfft=NFFT, num_src=1)
            doa_srp.locate_sources(X, freq_range=FREQS)
            if len(doa_srp.azimuth_recon) > 0:
                err_srp = np.degrees(np.abs(angular_error(doa_srp.azimuth_recon[0], true_doa_pra)))
        except: pass

        # MUSIC
        err_music = np.nan
        pred_music = np.nan
        try:
            doa_music = pra.doa.MUSIC(mic_pos, fs, nfft=NFFT, num_src=1)
            doa_music.locate_sources(X, freq_range=FREQS)
            if len(doa_music.azimuth_recon) > 0:
                pred_pra = doa_music.azimuth_recon[0]
                pred_music = (90 - np.degrees(pred_pra)) % 360
                err_music = np.degrees(np.abs(angular_error(pred_pra, true_doa_pra)))
        except: pass

        results.append({
            'sweep': row['sweep'],
            'clipped': is_clipped,
            'rms_avg': np.mean(rms),
            'pred_music': pred_music,
            'err_srp': err_srp,
            'err_music': err_music,
            'err_chip': np.degrees(np.abs(angular_error(np.radians((90-row['chip_doa_corrected'])%360), true_doa_pra)))
        })

    res_df = pd.DataFrame(results)
    
    print("\n--- Full Report for 0° Recordings ---")
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 1000)
    print(res_df[['sweep', 'clipped', 'rms_avg', 'pred_music', 'err_srp', 'err_music', 'err_chip']])
    
    print("\n--- Summary Statistics ---")
    print(res_df[['err_srp', 'err_music', 'err_chip']].describe())
    
    # Check if outliers are consistent across methods
    print("\n--- Consistency Check (Top 3 MUSIC errors) ---")
    print(res_df.nlargest(3, 'err_music')[['sweep', 'err_srp', 'err_music', 'err_chip']])

if __name__ == "__main__":
    analyze_0deg()
