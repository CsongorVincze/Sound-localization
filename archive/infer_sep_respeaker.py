import torch
import torchaudio
import numpy as np
import sounddevice as sd
import sys
import math
from models_sep import get_sep_model

# --- Configuration ---
# You can change this to 'unet' or 'tasnet' to test the other architectures!
MODEL_NAME = "fasnet" 
MODEL_PATH = f"best_{MODEL_NAME}_sep_soclas.pth"
SAMPLE_RATE = 16000
TARGET_SAMPLES = SAMPLE_RATE * 2 # 2 second buffer
BUFFER_SIZE = TARGET_SAMPLES

device = torch.device("cpu") # CPU is fine for edge inference

def main():
    print(f"Loading {MODEL_NAME.upper()} Separation & Localization model...")
    try:
        model = get_sep_model(MODEL_NAME)
        
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

    # Required for UNet
    spec_transform = torchaudio.transforms.Spectrogram(n_fft=512, win_length=512, hop_length=256, power=None).to(device)

    print("\n[!] Starting live audio stream from ReSpeaker...")
    print("[!] Stand in different corners and speak! Press Ctrl+C to stop.\n")

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
            if MODEL_NAME == 'unet':
                spec = spec_transform(mixed_wave)
                inputs = torch.view_as_real(spec).permute(0, 1, 4, 2, 3).reshape(1, 8, spec.size(2), spec.size(3))
            else:
                inputs = mixed_wave
                
            pred_audio, pred_doas = model(inputs)
            
        # Extract the two angles
        # pred_doas shape: (1, 2, 2) -> 2 sources, [sin, cos]
        doas = pred_doas[0].numpy()
        
        az1_rad = math.atan2(doas[0, 0], doas[0, 1])
        az2_rad = math.atan2(doas[1, 0], doas[1, 1])
        
        az1_deg = (math.degrees(az1_rad) + 360) % 360
        az2_deg = (math.degrees(az2_rad) + 360) % 360
        
        print(f"\r🎤 Tracking Source 1: {az1_deg:05.1f}°   |   🎤 Tracking Source 2: {az2_deg:05.1f}°     ", end="", flush=True)

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

if __name__ == "__main__":
    main()
