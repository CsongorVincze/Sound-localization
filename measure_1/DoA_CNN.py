"""
CNN-based DoA Estimation using GCC-PHAT Features
Uses PyTorch to train a CNN on GCC-PHAT correlation matrices.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import time

# =============================================================================
# CONFIGURATION
# =============================================================================
SAMPLE_RATE = 16000
MIC_SPACING = 0.0465
SPEED_OF_SOUND = 343.0
MAX_TAU = 2 * MIC_SPACING / SPEED_OF_SOUND
CC_LENGTH = 64  # Number of correlation samples to use

# Training config
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
TRAIN_SPLIT = 0.8

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =============================================================================
# GCC-PHAT FEATURE EXTRACTION
# =============================================================================

def compute_gcc_phat(sig1, sig2, fs, n_output=64):
    """
    Compute GCC-PHAT correlation vector between two signals.
    Returns a fixed-length correlation vector centered around zero lag.
    """
    n = len(sig1) + len(sig2)
    SIG1 = np.fft.rfft(sig1, n=n)
    SIG2 = np.fft.rfft(sig2, n=n)
    
    # Cross-spectrum with PHAT weighting
    R = SIG1 * np.conj(SIG2)
    mag = np.abs(R)
    R = R / (mag + 1e-10)
    
    # Inverse FFT to get correlation
    cc = np.fft.irfft(R, n=n)
    
    # Extract region around zero lag
    max_shift = int(MAX_TAU * fs) + n_output // 2
    cc = np.concatenate([cc[-max_shift:], cc[:max_shift+1]])
    
    # Center on zero lag and take n_output samples
    center = len(cc) // 2
    start = center - n_output // 2
    end = start + n_output
    
    return cc[start:end].astype(np.float32)

def extract_gcc_features(audio, fs=16000):
    """
    Extract GCC-PHAT features for all microphone pairs.
    Returns a 6xN matrix (6 mic pairs, N correlation samples).
    
    Mic pairs: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
    """
    n_mics = audio.shape[1]
    pairs = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
    
    features = []
    for i, j in pairs:
        cc = compute_gcc_phat(audio[:, i], audio[:, j], fs, CC_LENGTH)
        features.append(cc)
    
    return np.array(features, dtype=np.float32)  # Shape: (6, CC_LENGTH)

# =============================================================================
# SYNTHETIC DATA GENERATION
# =============================================================================

def generate_synthetic_sample(angle_deg, fs=16000, duration=0.1, snr_db=20):
    """
    Generate synthetic 4-microphone audio for a given DoA.
    """
    # Microphone positions (ReSpeaker v2.0 layout)
    mic_angles = np.radians([45, 315, 225, 135])
    mic_radius = MIC_SPACING / np.sqrt(2)
    mic_pos = np.array([
        [mic_radius * np.sin(a), mic_radius * np.cos(a)]
        for a in mic_angles
    ])
    
    # Direction vector (0° = front, clockwise)
    theta = np.radians(angle_deg)
    d = np.array([np.sin(theta), np.cos(theta)])
    
    # Calculate delays for each mic
    delays = -np.dot(mic_pos, d) / SPEED_OF_SOUND
    delays -= np.min(delays)  # Make all positive
    
    n_samples = int(duration * fs)
    t = np.arange(n_samples) / fs
    
    # Generate source signal (chirp for broadband)
    f0, f1 = 500, 3000
    signal = np.sin(2 * np.pi * (f0 + (f1-f0) * t / duration) * t)
    
    # Apply delays to each mic
    audio = np.zeros((n_samples, 4))
    for m in range(4):
        delay_samples = int(delays[m] * fs)
        if delay_samples < n_samples:
            audio[delay_samples:, m] = signal[:n_samples - delay_samples]
    
    # Add noise
    noise_power = np.var(signal) / (10 ** (snr_db / 10))
    audio += np.random.randn(*audio.shape) * np.sqrt(noise_power)
    
    return audio.astype(np.float32)

def generate_training_data(n_samples=5000, angles=None):
    """
    Generate synthetic training dataset.
    """
    if angles is None:
        angles = np.arange(0, 360, 5)  # 5° resolution
    
    X = []
    y = []
    
    print(f"Generating {n_samples} synthetic samples...")
    for i in range(n_samples):
        angle = np.random.choice(angles)
        snr = np.random.uniform(10, 30)  # Variable SNR
        
        audio = generate_synthetic_sample(angle, snr_db=snr)
        features = extract_gcc_features(audio)
        
        X.append(features)
        y.append(angle)
        
        if (i + 1) % 500 == 0:
            print(f"    Generated {i+1}/{n_samples}")
    
    return np.array(X), np.array(y)

# =============================================================================
# DATASET & MODEL
# =============================================================================

class DoADataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        # For regression: normalize to [0, 1] or use sin/cos
        # Using sin/cos representation to handle circular nature
        angles_rad = np.radians(y)
        self.y_sin = torch.from_numpy(np.sin(angles_rad)).float()
        self.y_cos = torch.from_numpy(np.cos(angles_rad)).float()
        self.y_deg = torch.from_numpy(y).float()
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y_sin[idx], self.y_cos[idx], self.y_deg[idx]

class DoACNN(nn.Module):
    """
    CNN for DoA estimation from GCC-PHAT features.
    Input: (batch, 6, 64) - 6 mic pairs, 64 correlation samples
    Output: (batch, 2) - sin(angle), cos(angle)
    """
    def __init__(self):
        super(DoACNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(6, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.3)
        
        # Fully connected layers
        # After 3 pooling: 64 -> 32 -> 16 -> 8
        self.fc1 = nn.Linear(128 * 8, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 2)  # Output: sin, cos
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x: (batch, 6, 64)
        x = self.pool(self.relu(self.bn1(self.conv1(x))))  # (batch, 32, 32)
        x = self.pool(self.relu(self.bn2(self.conv2(x))))  # (batch, 64, 16)
        x = self.pool(self.relu(self.bn3(self.conv3(x))))  # (batch, 128, 8)
        
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        
        # Normalize to unit circle
        x = x / (torch.norm(x, dim=1, keepdim=True) + 1e-8)
        
        return x

def angle_loss(pred, target_sin, target_cos):
    """
    Loss function for circular data.
    Uses cosine distance between predicted and target vectors.
    """
    target = torch.stack([target_sin, target_cos], dim=1)
    # Cosine similarity loss
    cos_sim = torch.sum(pred * target, dim=1)
    loss = 1 - cos_sim.mean()
    return loss

def pred_to_angle(sin_cos):
    """Convert sin/cos output to angle in degrees."""
    angles = np.arctan2(sin_cos[:, 0], sin_cos[:, 1])
    angles = np.degrees(angles)
    angles = (angles + 360) % 360
    return angles

# =============================================================================
# TRAINING
# =============================================================================

def train_model(model, train_loader, val_loader, epochs=100):
    """Train the CNN model."""
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    train_losses = []
    val_losses = []
    val_errors = []
    
    best_val_error = float('inf')
    best_model_state = None
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for X, y_sin, y_cos, y_deg in train_loader:
            X = X.to(DEVICE)
            y_sin = y_sin.to(DEVICE)
            y_cos = y_cos.to(DEVICE)
            
            optimizer.zero_grad()
            pred = model(X)
            loss = angle_loss(pred, y_sin, y_cos)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for X, y_sin, y_cos, y_deg in val_loader:
                X = X.to(DEVICE)
                y_sin = y_sin.to(DEVICE)
                y_cos = y_cos.to(DEVICE)
                
                pred = model(X)
                loss = angle_loss(pred, y_sin, y_cos)
                val_loss += loss.item()
                
                all_preds.append(pred.cpu().numpy())
                all_targets.append(y_deg.numpy())
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Calculate angle error
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        pred_angles = pred_to_angle(all_preds)
        
        errors = np.abs(pred_angles - all_targets)
        errors = np.minimum(errors, 360 - errors)  # Handle wraparound
        mean_error = np.mean(errors)
        val_errors.append(mean_error)
        
        scheduler.step(val_loss)
        
        if mean_error < best_val_error:
            best_val_error = mean_error
            best_model_state = model.state_dict().copy()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Error: {mean_error:.1f}°")
    
    # Restore best model
    model.load_state_dict(best_model_state)
    
    return train_losses, val_losses, val_errors

# =============================================================================
# COMPARISON WITH TRADITIONAL GCC-PHAT
# =============================================================================

def traditional_gcc_phat_angle(features, fs=16000):
    """
    Traditional GCC-PHAT DoA estimation from pre-computed features.
    Uses the correlation peaks to estimate TDOA.
    """
    # Features shape: (6, CC_LENGTH) for pairs (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
    # We'll use pairs (0,2) and (1,3) - diagonal pairs
    
    cc_02 = features[1]  # Pair (0,2)
    cc_13 = features[4]  # Pair (1,3)
    
    center = CC_LENGTH // 2
    
    # Find peaks
    peak_02 = np.argmax(cc_02) - center
    peak_13 = np.argmax(cc_13) - center
    
    # Convert to delays
    tau_02 = peak_02 / fs
    tau_13 = peak_13 / fs
    
    # TDOA to angle
    diag_dist = 2 * MIC_SPACING / np.sqrt(2)
    max_delay = diag_dist / SPEED_OF_SOUND
    
    d02 = np.clip(tau_02 / max_delay, -1, 1)
    d13 = np.clip(tau_13 / max_delay, -1, 1)
    
    c = 1 / np.sqrt(2)
    cos_theta = (d02 + d13) / (2 * c)
    sin_theta = (d02 - d13) / (2 * c)
    
    angle = np.degrees(np.arctan2(sin_theta, cos_theta))
    return (angle + 360) % 360

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print(" CNN DoA Estimator Training")
    print("=" * 60)
    print(f"    Device: {DEVICE}")
    
    # Generate or load data
    data_path = Path("doa_training_data.pkl")
    
    if data_path.exists():
        print("\n[1] Loading existing training data...")
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        X, y = data['X'], data['y']
    else:
        print("\n[1] Generating synthetic training data...")
        X, y = generate_training_data(n_samples=10000)
        with open(data_path, 'wb') as f:
            pickle.dump({'X': X, 'y': y}, f)
        print(f"    Saved to {data_path}")
    
    print(f"    Data shape: X={X.shape}, y={y.shape}")
    
    # Split data
    n_train = int(len(X) * TRAIN_SPLIT)
    indices = np.random.permutation(len(X))
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    
    print(f"    Train: {len(X_train)}, Validation: {len(X_val)}")
    
    # Create datasets and loaders
    train_dataset = DoADataset(X_train, y_train)
    val_dataset = DoADataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # Create model
    print("\n[2] Creating CNN model...")
    model = DoACNN().to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"    Total parameters: {total_params:,}")
    
    # Train
    print("\n[3] Training...")
    start_time = time.time()
    train_losses, val_losses, val_errors = train_model(model, train_loader, val_loader, EPOCHS)
    train_time = time.time() - start_time
    print(f"    Training completed in {train_time:.1f}s")
    
    # Save model
    torch.save(model.state_dict(), 'doa_cnn_model.pth')
    print("    Model saved: doa_cnn_model.pth")
    
    # Compare with traditional GCC-PHAT
    print("\n[4] Comparing with traditional GCC-PHAT...")
    model.eval()
    
    cnn_errors = []
    gcc_errors = []
    
    with torch.no_grad():
        for X_batch, y_sin, y_cos, y_deg in val_loader:
            X_batch = X_batch.to(DEVICE)
            
            # CNN prediction
            pred = model(X_batch)
            pred_angles = pred_to_angle(pred.cpu().numpy())
            
            # Traditional GCC-PHAT
            for i in range(len(X_batch)):
                features = X_batch[i].cpu().numpy()
                gcc_angle = traditional_gcc_phat_angle(features)
                
                true_angle = y_deg[i].item()
                
                cnn_err = abs(pred_angles[i] - true_angle)
                cnn_err = min(cnn_err, 360 - cnn_err)
                cnn_errors.append(cnn_err)
                
                gcc_err = abs(gcc_angle - true_angle)
                gcc_err = min(gcc_err, 360 - gcc_err)
                gcc_errors.append(gcc_err)
    
    print(f"\n    CNN Mean Error: {np.mean(cnn_errors):.1f}° (std: {np.std(cnn_errors):.1f}°)")
    print(f"    GCC-PHAT Mean Error: {np.mean(gcc_errors):.1f}° (std: {np.std(gcc_errors):.1f}°)")
    
    # Plotting
    print("\n[5] Generating plots...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("CNN DoA Estimator Performance", fontsize=16, fontweight='bold')
    
    # Plot 1: Training curves
    ax1 = axes[0, 0]
    ax1.plot(train_losses, label='Train Loss', color='blue')
    ax1.plot(val_losses, label='Val Loss', color='orange')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training & Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Validation error over epochs
    ax2 = axes[0, 1]
    ax2.plot(val_errors, color='green')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Mean Error (°)')
    ax2.set_title('Validation Error Over Training')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Error comparison
    ax3 = axes[1, 0]
    methods = ['CNN', 'GCC-PHAT']
    means = [np.mean(cnn_errors), np.mean(gcc_errors)]
    stds = [np.std(cnn_errors), np.std(gcc_errors)]
    colors = ['#00d2ff', '#ff6b6b']
    bars = ax3.bar(methods, means, yerr=stds, color=colors, alpha=0.8, capsize=5)
    for bar, m in zip(bars, means):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{m:.1f}°', ha='center', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Mean Error (°)')
    ax3.set_title('CNN vs Traditional GCC-PHAT')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Error distribution
    ax4 = axes[1, 1]
    ax4.hist(cnn_errors, bins=30, alpha=0.7, label='CNN', color='#00d2ff')
    ax4.hist(gcc_errors, bins=30, alpha=0.7, label='GCC-PHAT', color='#ff6b6b')
    ax4.set_xlabel('Error (°)')
    ax4.set_ylabel('Count')
    ax4.set_title('Error Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('cnn_doa_results.png', dpi=150)
    print("    Saved: cnn_doa_results.png")
    plt.show()
    
    print("\nDone!")

if __name__ == "__main__":
    main()
