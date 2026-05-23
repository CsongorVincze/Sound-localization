import torch
import torchaudio
import numpy as np
import sounddevice as sd
import sys
import math
import time
import matplotlib.pyplot as plt

try:
    import usb.core
    import usb.util
    HAS_USB = True
except ImportError:
    HAS_USB = False

sys.path.append('respeakeres_fileok')
try:
    from tuning import Tuning
    HAS_TUNING = True
except ImportError:
    HAS_TUNING = False

from models import get_model

# --- Configuration ---
# Choose from: 'tiny', 'mobilenet', 'transformer', 'resnet', 'waveform'
MODEL_NAME = "tiny" 
MODEL_PATH = f"best_{MODEL_NAME}_soclas.pth"
SAMPLE_RATE = 16000
TARGET_SAMPLES = SAMPLE_RATE * 2 # 2 second buffer
BUFFER_SIZE = TARGET_SAMPLES

device = torch.device("cpu") # CPU is fine for edge inference

def main():
    print(f"Loading {MODEL_NAME.upper()} Single-Source Localization model...")
    
    mic_tuning = None
    if HAS_USB and HAS_TUNING:
        try:
            dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
            if dev:
                mic_tuning = Tuning(dev)
                print("ReSpeaker USB Hardware Tuning initialized for DoA extraction.")
            else:
                print("Warning: ReSpeaker USB device not found for Hardware DoA.")
        except Exception as e:
            print(f"Warning: Failed to initialize USB device: {e}")

    try:
        # Get the blueprint from our original models.py
        model = get_model(MODEL_NAME)
        
        # Load weights (fixing the DataParallel 'module.' prefix if it was trained on multi-GPU)
        state_dict = torch.load(MODEL_PATH, map_location=device)
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
            
        model.load_state_dict(new_state_dict)
        model.eval()
        print("Model loaded successfully!")
    except FileNotFoundError:
        print(f"Error: Could not find '{MODEL_PATH}'. Make sure you downloaded it.")
        sys.exit(1)

    # Required for the spectrogram models
    spec_transform = torchaudio.transforms.Spectrogram(n_fft=512, win_length=512, hop_length=256, power=None).to(device)

    print("\n[!] Starting live audio stream from ReSpeaker...")
    print("[!] Stand in different corners and speak! Press Ctrl+C to stop.\n")

    start_time = time.time()
    log_times = []
    log_model_doa = []
    log_hw_doa = []

    def audio_callback(indata, frames, time_info, status):
        if status:
            pass
            
        # ReSpeaker v2 usually outputs 6 channels via USB. We ONLY want the first 4 (the raw mics).
        if indata.shape[1] >= 4:
            audio_buffer = indata[:, :4]
        else:
            audio_buffer = indata

        waveform = torch.tensor(audio_buffer.T, dtype=torch.float32)
        waveform = waveform / (waveform.abs().max() + 1e-8)
        
        # Force exactly TARGET_SAMPLES
        if waveform.shape[1] > TARGET_SAMPLES:
            waveform = waveform[:, :TARGET_SAMPLES]
        elif waveform.shape[1] < TARGET_SAMPLES:
            waveform = torch.nn.functional.pad(waveform, (0, TARGET_SAMPLES - waveform.shape[1]))

        mixed_wave = waveform.unsqueeze(0) # (1, 4, T)

        # Run Inference!
        with torch.no_grad():
            if MODEL_NAME != 'waveform':
                spec = spec_transform(mixed_wave)
                # Correctly reshape 5D complex spectrogram into 8 channels
                inputs = torch.view_as_real(spec).permute(0, 1, 4, 2, 3).reshape(1, 8, spec.size(2), spec.size(3))
            else:
                inputs = mixed_wave
                
            pred_doas = model(inputs)
            
        # Extract the single angle
        # pred_doas shape: (1, 2) -> [sin, cos]
        doas = pred_doas[0].numpy()
        
        az_rad = math.atan2(doas[0], doas[1])
        az_deg = (math.degrees(az_rad) + 360) % 360
        
        hw_doa_val = None
        hw_doa_str = "N/A"
        if mic_tuning:
            try:
                hw_doa_val = mic_tuning.direction
                hw_doa_str = f"{hw_doa_val:03d}°"
            except Exception:
                pass
                
        print(f"\r🎤 Model DoA: {az_deg:05.1f}° | Hardware DoA: {hw_doa_str}              ", end="", flush=True)
        
        log_times.append(time.time() - start_time)
        log_model_doa.append(az_deg)
        log_hw_doa.append(hw_doa_val)

    try:
        # Request 6 channels (Standard ReSpeaker USB hardware profile)
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=6, blocksize=BUFFER_SIZE, callback=audio_callback):
            while True:
                sd.sleep(100)
    except KeyboardInterrupt:
        print("\nStopped.")
    except Exception as e:
        print(f"\n[X] Error opening audio stream: {e}")
        print("Tip: Make sure you run this on the LOCAL laptop where the ReSpeaker is physically plugged in!")

    print("\nGenerating plot...")
    if len(log_times) > 0:
        plt.figure(figsize=(10, 6))
        plt.plot(log_times, log_model_doa, label='Model DoA', marker='o', linestyle='-', alpha=0.8)
        
        hw_valid_times = [t for t, val in zip(log_times, log_hw_doa) if val is not None]
        hw_valid_vals = [val for val in log_hw_doa if val is not None]
        
        if hw_valid_times:
            plt.plot(hw_valid_times, hw_valid_vals, label='Hardware DoA', marker='x', linestyle='--', alpha=0.8)
            
        plt.xlabel('Time (s)')
        plt.ylabel('Direction of Arrival (°)')
        plt.title('DoA Estimation Over Time')
        plt.legend()
        plt.grid(True)
        plt.yticks(np.arange(0, 361, 45))
        plt.ylim(0, 360)
        plt.tight_layout()
        plt.show()
    else:
        print("No data to plot.")

if __name__ == "__main__":
    main()
