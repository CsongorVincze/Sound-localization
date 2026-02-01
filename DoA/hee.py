"""
ReSpeaker v2.0 - Software DoA Calculation using GCC-PHAT
Calculates Direction of Arrival from raw 4-channel microphone data
"""
import sounddevice as sd
import numpy as np
from scipy import signal
import queue
import time
import sys

# === CONFIGURATION ===
RATE = 16000           # Sample rate
CHUNK_SIZE = 1024      # Samples per chunk
SPEED_OF_SOUND = 343.0 # m/s at room temperature

# ReSpeaker v2.0 Microphone Geometry
# 4 mics in a circle, diameter ~58mm
# Mic positions (in meters): 
#   Mic 0: 0° (front)
#   Mic 1: 90° (right) 
#   Mic 2: 180° (back)
#   Mic 3: 270° (left)
MIC_DISTANCE = 0.058  # 58mm diameter
MIC_RADIUS = MIC_DISTANCE / 2

# Microphone angles in radians (counterclockwise from front)
MIC_ANGLES = np.array([0, np.pi/2, np.pi, 3*np.pi/2])

# Queue for audio
audio_queue = queue.Queue()

def gcc_phat(sig1, sig2, fs, max_tau=None):
    """
    Generalized Cross-Correlation with Phase Transform
    Returns the time delay (in samples) between sig1 and sig2
    
    Args:
        sig1, sig2: Audio signals from two microphones
        fs: Sample rate
        max_tau: Maximum delay to search (in seconds)
    
    Returns:
        tau: Time delay in seconds (positive = sig2 leads sig1)
    """
    n = len(sig1) + len(sig2)
    
    # FFT of both signals
    SIG1 = np.fft.rfft(sig1, n=n)
    SIG2 = np.fft.rfft(sig2, n=n)
    
    # Cross-spectrum with phase transform (whitening)
    R = SIG1 * np.conj(SIG2)
    R = R / (np.abs(R) + 1e-10)  # Normalize (PHAT weighting)
    
    # Inverse FFT to get cross-correlation
    cc = np.fft.irfft(R, n=n)
    
    # Find the peak
    if max_tau is not None:
        max_shift = int(max_tau * fs)
    else:
        max_shift = len(sig1) // 2
    
    # Search in valid range [-max_shift, max_shift]
    cc = np.concatenate([cc[-max_shift:], cc[:max_shift+1]])
    
    # Find peak
    peak_idx = np.argmax(np.abs(cc))
    tau_samples = peak_idx - max_shift
    
    # Convert to seconds
    tau = tau_samples / fs
    
    return tau

def estimate_doa(audio_chunk, fs):
    """
    Estimate Direction of Arrival using multiple microphone pairs
    
    Args:
        audio_chunk: (samples, channels) array with 4+ channels
        fs: Sample rate
    
    Returns:
        angle: Estimated angle in degrees (0-360)
        confidence: Confidence score (0-1)
    """
    # Extract microphone channels (assuming channels 0-3 are the 4 mics)
    # For ReSpeaker 6-channel: Channel 0 is processed, 1-4 are raw mics, 5 is loopback
    # We use channels 1,2,3,4 (indices 0,1,2,3 if only 4 channels)
    
    n_channels = audio_chunk.shape[1]
    
    if n_channels >= 6:
        # 6-channel firmware: Use channels 1-4 (raw mics)
        mics = audio_chunk[:, 1:5].astype(np.float64)
    elif n_channels >= 4:
        # 4-channel: Use all
        mics = audio_chunk[:, 0:4].astype(np.float64)
    else:
        return 0, 0  # Not enough channels
    
    # Max delay based on mic distance
    max_tau = MIC_DISTANCE / SPEED_OF_SOUND * 2
    
    # Calculate delays between opposite mic pairs
    # Pair 1: Mic 0 (front) vs Mic 2 (back) -> front-back axis
    # Pair 2: Mic 1 (right) vs Mic 3 (left) -> left-right axis
    
    tau_02 = gcc_phat(mics[:, 0], mics[:, 2], fs, max_tau)  # Front-Back
    tau_13 = gcc_phat(mics[:, 1], mics[:, 3], fs, max_tau)  # Right-Left
    
    # Convert delays to angle
    # For opposite mics, delay = (d * cos(theta)) / c
    # where d is mic distance, theta is angle from mic axis
    
    # Normalize delays to [-1, 1] range
    max_delay = MIC_DISTANCE / SPEED_OF_SOUND
    
    sin_theta1 = np.clip(tau_02 / max_delay, -1, 1)
    sin_theta2 = np.clip(tau_13 / max_delay, -1, 1)
    
    # Calculate angle using atan2
    # tau_02 gives front-back component (y-axis)
    # tau_13 gives left-right component (x-axis)
    angle_rad = np.arctan2(sin_theta1, sin_theta2)
    
    # Convert to degrees (0-360)
    angle_deg = np.degrees(angle_rad)
    if angle_deg < 0:
        angle_deg += 360
    
    # Confidence based on signal energy and correlation strength
    energy = np.mean(mics**2)
    confidence = min(1.0, energy / 1000000)  # Arbitrary threshold
    
    return angle_deg, confidence

def audio_callback(indata, frames, time_info, status):
    if status:
        print(status, file=sys.stderr)
    audio_queue.put(indata.copy())

def find_respeaker():
    """Find ReSpeaker audio device."""
    for i, dev in enumerate(sd.query_devices()):
        if dev['max_input_channels'] > 0:
            name = dev['name'].lower()
            if 'respeaker' in name or 'uac1.0' in name:
                return i, dev['max_input_channels'], dev['name']
    return None, None, None

def main():
    print("=" * 60)
    print(" ReSpeaker DoA - Software Calculation (GCC-PHAT)")
    print("=" * 60)
    
    # Find audio device
    print("\nSearching for ReSpeaker audio device...")
    dev_idx, channels, dev_name = find_respeaker()
    
    if dev_idx is None:
        print("ERROR: ReSpeaker audio device not found!")
        return
    
    print(f"  Device: {dev_name}")
    print(f"  Channels: {channels}")
    
    if channels < 4:
        print("ERROR: Need at least 4 channels for DoA calculation!")
        return
    
    # Start streaming
    print("\n" + "=" * 60)
    print(" RUNNING - Move sound source around the microphone array!")
    print(" The angle indicates the direction of the sound source.")
    print("=" * 60 + "\n")
    
    # Smoothing for display
    angle_history = []
    history_size = 5
    
    with sd.InputStream(device=dev_idx, channels=channels, samplerate=RATE,
                        blocksize=CHUNK_SIZE, callback=audio_callback, 
                        dtype='int16'):
        
        try:
            while True:
                # Get audio chunk
                data = audio_queue.get()
                
                # Calculate DoA
                angle, confidence = estimate_doa(data, RATE)
                
                # Smooth the angle (circular averaging)
                angle_history.append(angle)
                if len(angle_history) > history_size:
                    angle_history.pop(0)
                
                # Circular mean
                angles_rad = np.radians(angle_history)
                mean_sin = np.mean(np.sin(angles_rad))
                mean_cos = np.mean(np.cos(angles_rad))
                smooth_angle = np.degrees(np.arctan2(mean_sin, mean_cos))
                if smooth_angle < 0:
                    smooth_angle += 360
                
                # Audio level
                rms = np.sqrt(np.mean(data.astype(float)**2))
                bar_len = min(int(rms / 300), 40)
                bar = "█" * bar_len
                
                # Direction indicator
                direction = ""
                if 337.5 <= smooth_angle or smooth_angle < 22.5:
                    direction = "FRONT"
                elif 22.5 <= smooth_angle < 67.5:
                    direction = "FRONT-RIGHT"
                elif 67.5 <= smooth_angle < 112.5:
                    direction = "RIGHT"
                elif 112.5 <= smooth_angle < 157.5:
                    direction = "BACK-RIGHT"
                elif 157.5 <= smooth_angle < 202.5:
                    direction = "BACK"
                elif 202.5 <= smooth_angle < 247.5:
                    direction = "BACK-LEFT"
                elif 247.5 <= smooth_angle < 292.5:
                    direction = "LEFT"
                else:
                    direction = "FRONT-LEFT"
                
                # Print status
                print(f"DoA: {smooth_angle:5.1f}° ({direction:12s}) | {bar.ljust(40)}", end='\r')
                
        except KeyboardInterrupt:
            print("\n\nStopped.")

if __name__ == "__main__":
    main()