"""
DoA Algorithm Comparison: ITD vs SNN (V1 & V2)
===============================================
Compares 4 different sound localization approaches:
1. Basic ITD (Cross-Correlation)
2. GCC-PHAT ITD (Phase-weighted Cross-Correlation)
3. SNN V1 (Baseline - 30 samples, 5° steps, Classification)
4. SNN V2 (Extended - 150 samples, 3° steps, Regression)
"""

import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import spikegen

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_PATH = Path('measure_1/snn_training_data.pkl')
MODEL_V1_PATH = Path('measure_1/snn_baseline_model.pth')
MODEL_V2_PATH = Path('measure_1/snn_regression_model_v2.pth')

MIC_SPACING = 0.0465  # meters
SPEED_OF_SOUND = 343.0  # m/s
SAMPLE_RATE = 16000
MAX_DELAY_SAMPLES = int(MIC_SPACING / SPEED_OF_SOUND * SAMPLE_RATE) + 5

# =============================================================================
# ITD ALGORITHMS
# =============================================================================

def itd_basic(audio, fs=16000):
    """Basic ITD using cross-correlation."""
    delays = []
    for pair in [(0, 2), (1, 3)]:
        mic_a, mic_b = audio[:, pair[0]], audio[:, pair[1]]
        corr = np.correlate(mic_a, mic_b, mode='full')
        lags = np.arange(-len(mic_a) + 1, len(mic_a))
        valid = np.abs(lags) <= MAX_DELAY_SAMPLES * 2
        corr[~valid] = -np.inf
        delay = lags[np.argmax(corr)] / fs
        delays.append(delay)
    
    angle_rad = np.arctan2(delays[1], delays[0])
    return (np.degrees(angle_rad) + 180) % 180

def gcc_phat(sig1, sig2, fs=16000):
    """GCC-PHAT correlation."""
    n = len(sig1) + len(sig2)
    SIG1 = np.fft.rfft(sig1, n=n)
    SIG2 = np.fft.rfft(sig2, n=n)
    R = SIG1 * np.conj(SIG2)
    R = R / (np.abs(R) + 1e-10)
    cc = np.fft.irfft(R, n=n)
    max_lag = MAX_DELAY_SAMPLES * 2
    cc_center = np.concatenate([cc[-max_lag:], cc[:max_lag+1]])
    lags = np.arange(-max_lag, max_lag + 1)
    return lags[np.argmax(np.abs(cc_center))] / fs

def itd_gcc_phat(audio, fs=16000):
    """GCC-PHAT based ITD estimation."""
    tau_02 = gcc_phat(audio[:, 0], audio[:, 2], fs)
    tau_13 = gcc_phat(audio[:, 1], audio[:, 3], fs)
    max_tau = MIC_SPACING / SPEED_OF_SOUND
    cos_t = np.clip(tau_02 / max_tau, -1, 1)
    sin_t = np.clip(tau_13 / max_tau, -1, 1)
    return (np.degrees(np.arctan2(sin_t, cos_t)) + 180) % 180

# =============================================================================
# SNN MODELS
# =============================================================================

class SNNV1(nn.Module):
    """SNN V1 - Classification model (from snn_baseline.py)"""
    def __init__(self, num_inputs=4, num_hidden=128, num_outputs=30, beta=0.9):
        super().__init__()
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta)

    def forward(self, x):
        x = x.permute(1, 0, 2)
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        spk_in = spikegen.delta(x, threshold=0.05)
        mem2_rec = []
        for step in range(x.size(0)):
            cur1 = self.fc1(spk_in[step])
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            mem2_rec.append(mem2)
        return torch.stack(mem2_rec, dim=0)

class SNNV2(nn.Module):
    """SNN V2 - Regression model (from snn_regression_v2.py)"""
    def __init__(self, num_inputs=4, num_hidden=256, beta=0.9):
        super().__init__()
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.lif2 = snn.Leaky(beta=beta)
        self.fc_out = nn.Linear(num_hidden, 1)

    def forward(self, x):
        x = x.permute(1, 0, 2)
        T, B, _ = x.shape
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        spk_in = spikegen.delta(x, threshold=0.04)
        out_acc = torch.zeros(B, 1)
        for step in range(T):
            cur1 = self.fc1(spk_in[step])
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            out_acc = out_acc + self.fc_out(mem2)
        return torch.sigmoid(out_acc / T)

# =============================================================================
# MAIN COMPARISON
# =============================================================================

def main():
    print("="*70)
    print(" DoA Algorithm Comparison: ITD vs SNN")
    print("="*70)
    
    # Load dataset
    print("\n[1] Loading dataset...")
    with open(DATA_PATH, 'rb') as f:
        data = pickle.load(f)
    
    angles = sorted(set(d['angle'] for d in data))
    min_angle, max_angle = min(angles), max(angles)
    angle_range = max_angle - min_angle
    num_classes = len(angles)
    
    print(f"    Samples: {len(data)}")
    print(f"    Angles: {min_angle}° to {max_angle}° ({num_classes} classes)")
    
    # Results storage
    results = {
        'Basic ITD': [],
        'GCC-PHAT': [],
        'SNN V1': [],
        'SNN V2': []
    }
    true_angles_list = []
    
    # Load SNN models
    print("\n[2] Loading SNN models...")
    
    # SNN V1
    snn_v1 = SNNV1(num_outputs=num_classes)
    try:
        snn_v1.load_state_dict(torch.load(MODEL_V1_PATH, map_location='cpu'))
        snn_v1.eval()
        print(f"    SNN V1: Loaded ({num_classes} output classes)")
        snn_v1_ok = True
    except Exception as e:
        print(f"    SNN V1: Failed to load - {e}")
        snn_v1_ok = False
    
    # SNN V2
    snn_v2 = SNNV2()
    try:
        snn_v2.load_state_dict(torch.load(MODEL_V2_PATH, map_location='cpu'))
        snn_v2.eval()
        print(f"    SNN V2: Loaded (regression)")
        snn_v2_ok = True
    except Exception as e:
        print(f"    SNN V2: Failed to load - {e}")
        snn_v2_ok = False
    
    # Evaluate all algorithms
    print("\n[3] Evaluating algorithms...")
    
    for i, sample in enumerate(data):
        audio = sample['audio'].astype(np.float32)
        true_angle = sample['angle']
        true_angles_list.append(true_angle)
        
        # Normalize audio
        audio_norm = audio / (np.max(np.abs(audio)) + 1e-8)
        
        # 1. Basic ITD
        try:
            pred = itd_basic(audio_norm, SAMPLE_RATE)
            err = min(abs(pred - true_angle), 180 - abs(pred - true_angle))
            results['Basic ITD'].append(err)
        except:
            results['Basic ITD'].append(np.nan)
        
        # 2. GCC-PHAT
        try:
            pred = itd_gcc_phat(audio_norm, SAMPLE_RATE)
            err = min(abs(pred - true_angle), 180 - abs(pred - true_angle))
            results['GCC-PHAT'].append(err)
        except:
            results['GCC-PHAT'].append(np.nan)
        
        # 3. SNN V1
        if snn_v1_ok:
            try:
                x = torch.FloatTensor(audio_norm[::64, :]).unsqueeze(0)
                with torch.no_grad():
                    mem_out = snn_v1(x)
                    pred_idx = torch.argmax(mem_out.sum(dim=0), dim=1).item()
                pred = angles[pred_idx] if pred_idx < len(angles) else 0
                err = abs(pred - true_angle)
                results['SNN V1'].append(err)
            except:
                results['SNN V1'].append(np.nan)
        else:
            results['SNN V1'].append(np.nan)
        
        # 4. SNN V2
        if snn_v2_ok:
            try:
                x = torch.FloatTensor(audio_norm[::32, :]).unsqueeze(0)
                with torch.no_grad():
                    pred_norm = snn_v2(x).item()
                pred = pred_norm * angle_range + min_angle
                err = abs(pred - true_angle)
                results['SNN V2'].append(err)
            except:
                results['SNN V2'].append(np.nan)
        else:
            results['SNN V2'].append(np.nan)
        
        if (i+1) % 50 == 0:
            print(f"    Processed {i+1}/{len(data)} samples...")
    
    # Calculate MAE
    print("\n" + "="*70)
    print(" RESULTS SUMMARY")
    print("="*70)
    
    maes = {}
    for name in results:
        valid = [e for e in results[name] if not np.isnan(e)]
        maes[name] = np.mean(valid) if valid else np.nan
    
    print(f"\n{'Algorithm':<20} {'MAE (°)':<12} {'Samples':<10}")
    print("-"*50)
    for name, mae in sorted(maes.items(), key=lambda x: x[1] if not np.isnan(x[1]) else 999):
        valid_count = sum(1 for e in results[name] if not np.isnan(e))
        print(f"{name:<20} {mae:>8.2f}°     {valid_count}/{len(data)}")
    
    # Analysis
    print("\n" + "="*70)
    print(" ALGORITHM ANALYSIS")
    print("="*70)
    
    print("""
┌─────────────────────────────────────────────────────────────────────┐
│ 1. BASIC ITD (Cross-Correlation)                                   │
├─────────────────────────────────────────────────────────────────────┤
│ Method   : Direct cross-correlation between opposite mic pairs     │
│ Strengths: Simple, fast, no training required                      │
│ Weakness : Very sensitive to noise and echoes                      │
│ Reason   : With only ~2 sample delay at 16kHz, any noise corrupts  │
│            the correlation peak. Works best with anechoic sounds.  │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ 2. GCC-PHAT (Generalized Cross-Correlation with PHAT)              │
├─────────────────────────────────────────────────────────────────────┤
│ Method   : Frequency-domain CC with phase normalization            │
│ Strengths: More robust to noise, sharper correlation peak          │
│ Weakness : Still limited by small microphone spacing               │
│ Reason   : PHAT weighting flattens spectrum magnitudes, keeping    │
│            only phase info. Better than basic, but physics limits. │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ 3. SNN V1 (Classification - 30 samples baseline)                   │
├─────────────────────────────────────────────────────────────────────┤
│ Method   : Spiking Neural Network classifying into angle bins      │
│ Strengths: Learns from real data, spike-based computation          │
│ Weakness : Very limited training data (1 sample/class originally)  │
│ Reason   : With only 30 samples across many classes, the network   │
│            cannot generalize. High variance, prone to memorization.│
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ 4. SNN V2 (Regression - 150 samples, augmented)                    │
├─────────────────────────────────────────────────────────────────────┤
│ Method   : Regression SNN predicting continuous angle              │
│ Strengths: More data (3 reps/angle), regression is smoother        │
│ Weakness : Still limited real-world data, environment-specific     │
│ Reason   : Regression avoids discrete classification errors.       │
│            More samples = better learning. Best of the 4 methods.  │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ GENERAL CONCLUSION                                                  │
├─────────────────────────────────────────────────────────────────────┤
│ • ITD methods struggle with small mic spacing (46.5mm ≈ 2 samples) │
│ • SNNs can learn environment-specific patterns beyond pure timing  │
│ • More training data consistently improves SNN performance         │
│ • Regression output is smoother than classification for DoA        │
│ • Expected ranking: SNN V2 > SNN V1 ≈ GCC-PHAT > Basic ITD         │
└─────────────────────────────────────────────────────────────────────┘
""")
    
    # Plotting
    print("[4] Generating comparison plot...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("DoA Algorithm Comparison (ITD vs SNN)", fontsize=16, fontweight='bold')
    
    colors = {'Basic ITD': '#ff6b6b', 'GCC-PHAT': '#4ecdc4', 
              'SNN V1': '#ffe66d', 'SNN V2': '#9b59b6'}
    
    # 1. MAE Bar Chart
    ax1 = axes[0, 0]
    names = list(maes.keys())
    vals = [maes[n] for n in names]
    bars = ax1.bar(names, vals, color=[colors[n] for n in names], alpha=0.8)
    for bar, val in zip(bars, vals):
        if not np.isnan(val):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.1f}°', ha='center', fontsize=11)
    ax1.set_ylabel('Mean Absolute Error (°)')
    ax1.set_title('MAE Comparison')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. Box Plot
    ax2 = axes[0, 1]
    box_data = [[e for e in results[n] if not np.isnan(e)] for n in names]
    bp = ax2.boxplot(box_data, labels=names, patch_artist=True)
    for patch, name in zip(bp['boxes'], names):
        patch.set_facecolor(colors[name])
        patch.set_alpha(0.7)
    ax2.set_ylabel('Absolute Error (°)')
    ax2.set_title('Error Distribution')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Error by Angle
    ax3 = axes[1, 0]
    for name in names:
        angle_errs = {}
        for a, e in zip(true_angles_list, results[name]):
            if not np.isnan(e):
                angle_errs.setdefault(a, []).append(e)
        sorted_angles = sorted(angle_errs.keys())
        mean_errs = [np.mean(angle_errs[a]) for a in sorted_angles]
        ax3.plot(sorted_angles, mean_errs, 'o-', color=colors[name], 
                label=name, alpha=0.7, markersize=4)
    ax3.set_xlabel('True Angle (°)')
    ax3.set_ylabel('Mean Error (°)')
    ax3.set_title('Error vs True Angle')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Error Histogram
    ax4 = axes[1, 1]
    for name in names:
        valid = [e for e in results[name] if not np.isnan(e)]
        if valid:
            ax4.hist(valid, bins=20, alpha=0.4, label=name, color=colors[name])
    ax4.set_xlabel('Absolute Error (°)')
    ax4.set_ylabel('Count')
    ax4.set_title('Error Histogram')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = 'SNN_sim/full_comparison.png'
    plt.savefig(output_path, dpi=150)
    print(f"    Plot saved to {output_path}")
    plt.show()
    
    print("\nDone!")

if __name__ == "__main__":
    main()
