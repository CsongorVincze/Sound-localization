"""
DoA Algorithms for ReSpeaker v2.0
Extracted from the working DoA comparison code
"""
import numpy as np

# ReSpeaker v2.0 Configuration
SPEED_OF_SOUND = 343.0
MIC_SPACING = 0.0465  # 46.5mm between adjacent mics

# Microphone angles (CLOCKWISE from FRONT = 0°)
MIC_ANGLE_DEG = np.array([45, 315, 225, 135])  # M0, M1, M2, M3
MIC_RADIUS = MIC_SPACING / np.sqrt(2)

# Microphone positions in Cartesian (x=RIGHT, y=FRONT)
MIC_POSITIONS = np.array([
    [MIC_RADIUS * np.sin(np.radians(MIC_ANGLE_DEG[i])), 
     MIC_RADIUS * np.cos(np.radians(MIC_ANGLE_DEG[i]))]
    for i in range(4)
])

def gcc_phat_delay(sig1, sig2, fs, max_tau):
    """
    GCC-PHAT to find time delay between two signals.
    Returns delay in seconds (positive if sig2 arrives before sig1).
    """
    n = len(sig1) + len(sig2)
    SIG1 = np.fft.rfft(sig1, n=n)
    SIG2 = np.fft.rfft(sig2, n=n)
    
    # Apply human voice bandpass filter (300Hz - 3400Hz)
    freqs = np.fft.rfftfreq(n, d=1/fs)
    voice_mask = (freqs >= 300) & (freqs <= 3400)
    SIG1[~voice_mask] = 0
    SIG2[~voice_mask] = 0
    
    # Cross-spectrum with modified PHAT weighting
    R = SIG1 * np.conj(SIG2)
    mag = np.abs(R)
    beta = 0.2 * np.max(mag)  # Increased regularization for speech harmonics
    R = R / (mag + beta + 1e-10)
    
    cc = np.fft.irfft(R, n=n)
    
    max_shift = int(max_tau * fs)
    cc = np.concatenate([cc[-max_shift:], cc[:max_shift+1]])
    
    # Find peak with parabolic interpolation
    peak_idx = np.argmax(np.abs(cc))
    
    if 1 <= peak_idx <= len(cc) - 2:
        y0, y1, y2 = np.abs(cc[peak_idx-1:peak_idx+2])
        if y0 != 2*y1 - y2:
            delta = 0.5 * (y0 - y2) / (y0 - 2*y1 + y2)
            peak_idx = peak_idx + delta
    
    tau_samples = peak_idx - max_shift
    return tau_samples / fs

def tdoa_to_angle(tau_02, tau_13):
    """
    Convert TDOA from two diagonal mic pairs to angle.
    
    Returns angle in degrees: 0° = FRONT, clockwise positive
    """
    # Maximum possible delay for diagonal pairs
    diag_dist = 2 * MIC_RADIUS
    max_delay = diag_dist / SPEED_OF_SOUND
    
    # Normalize to [-1, 1]
    d02 = np.clip(tau_02 / max_delay, -1, 1)
    d13 = np.clip(tau_13 / max_delay, -1, 1)
    
    # Convert delays to angle components
    c = 1 / np.sqrt(2)
    cos_theta = (d02 + d13) / (2 * c)
    sin_theta = (d02 - d13) / (2 * c)
    
    # Clip to valid range
    cos_theta = np.clip(cos_theta, -1, 1)
    sin_theta = np.clip(sin_theta, -1, 1)
    
    # Calculate angle
    angle_rad = np.arctan2(sin_theta, cos_theta)
    angle_deg = np.degrees(angle_rad)
    
    return (angle_deg + 360) % 360

def get_gcc_phat_angle(mics, fs=16000):
    """
    Calculate DoA using GCC-PHAT algorithm.
    
    Args:
        mics: Audio data with shape (samples, 4) for 4 microphones
        fs: Sample rate (default 16000 Hz)
    
    Returns:
        angle: Direction of arrival in degrees (0° = FRONT, clockwise)
    """
    max_tau = 2 * MIC_SPACING / SPEED_OF_SOUND
    
    # Use opposite mic pairs
    # M0 (45°, front-right) vs M2 (225°, back-left)
    # M1 (315°, front-left) vs M3 (135°, back-right)
    tau_02 = gcc_phat_delay(mics[:, 0], mics[:, 2], fs, max_tau)
    tau_13 = gcc_phat_delay(mics[:, 1], mics[:, 3], fs, max_tau)
    
    return tdoa_to_angle(tau_02, tau_13)

def get_srp_phat_angle(mics, fs=16000, resolution=5):
    """
    Calculate DoA using SRP-PHAT algorithm.
    
    Args:
        mics: Audio data with shape (samples, 4) for 4 microphones
        fs: Sample rate (default 16000 Hz)
        resolution: Angular resolution in degrees (default 5°)
    
    Returns:
        angle: Direction of arrival in degrees (0° = FRONT, clockwise)
    """
    angles = np.arange(0, 360, resolution)
    powers = np.zeros(len(angles))
    
    n_mics = mics.shape[1]
    n_samples = mics.shape[0]
    
    # FFT of all channels
    n_fft = 2 * n_samples
    freqs = np.fft.rfftfreq(n_fft, 1/fs)
    MICs = np.array([np.fft.rfft(mics[:, i], n=n_fft) for i in range(n_mics)])
    
    # Apply human voice bandpass filter (300Hz - 3400Hz)
    voice_mask = (freqs >= 300) & (freqs <= 3400)
    for m in range(n_mics):
        MICs[m, ~voice_mask] = 0
        
    for idx, angle in enumerate(angles):
        theta = np.radians(angle)
        # Direction vector: x=sin(θ), y=cos(θ)
        d = np.array([np.sin(theta), np.cos(theta)])
        
        # Calculate delays
        delays = -np.dot(MIC_POSITIONS, d) / SPEED_OF_SOUND
        
        # Steer and sum
        steered_sum = np.zeros(len(freqs), dtype=complex)
        for m in range(n_mics):
            phase_shift = np.exp(2j * np.pi * freqs * delays[m])
            mag = np.abs(MICs[m])
            weighted = MICs[m] / (mag + 0.2 * np.max(mag) + 1e-10)  # Regularized PHAT for voice
            steered_sum += weighted * phase_shift
        
        powers[idx] = np.sum(np.abs(steered_sum)**2)
    
    return angles[np.argmax(powers)]

def basic_cc_delay(sig1, sig2, fs, max_tau):
    """Basic cross-correlation without PHAT weighting."""
    n = len(sig1) + len(sig2)
    SIG1 = np.fft.rfft(sig1, n=n)
    SIG2 = np.fft.rfft(sig2, n=n)
    
    # Apply human voice bandpass filter (300Hz - 3400Hz)
    freqs = np.fft.rfftfreq(n, d=1/fs)
    voice_mask = (freqs >= 300) & (freqs <= 3400)
    SIG1[~voice_mask] = 0
    SIG2[~voice_mask] = 0
    
    R = SIG1 * np.conj(SIG2)
    cc = np.fft.irfft(R, n=n)
    
    max_shift = int(max_tau * fs)
    cc = np.concatenate([cc[-max_shift:], cc[:max_shift+1]])
    
    peak_idx = np.argmax(np.abs(cc))
    
    # Parabolic interpolation
    if 1 <= peak_idx <= len(cc) - 2:
        y0, y1, y2 = np.abs(cc[peak_idx-1:peak_idx+2])
        if y0 != 2*y1 - y2:
            delta = 0.5 * (y0 - y2) / (y0 - 2*y1 + y2)
            peak_idx = peak_idx + delta
    
    tau_samples = peak_idx - max_shift
    return tau_samples / fs

def get_basic_cc_angle(mics, fs=16000):
    """DoA using basic cross-correlation."""
    max_tau = 2 * MIC_SPACING / SPEED_OF_SOUND
    
    tau_02 = basic_cc_delay(mics[:, 0], mics[:, 2], fs, max_tau)
    tau_13 = basic_cc_delay(mics[:, 1], mics[:, 3], fs, max_tau)
    
    return tdoa_to_angle(tau_02, tau_13)












