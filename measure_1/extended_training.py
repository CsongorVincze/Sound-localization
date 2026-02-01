"""
Extended CNN Training with Diverse Sound Types
Collects more data with varied sounds for robust DoA estimation.
"""
import serial
import serial.tools.list_ports
import time
import sounddevice as sd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from pathlib import Path
import pickle

# =============================================================================
# CONFIGURATION
# =============================================================================
COM_PORT = None
BAUD_RATE = 9600
SAMPLE_RATE = 16000
RECORDING_DURATION = 1.0
MIC_SPACING = 0.0465
SPEED_OF_SOUND = 343.0
MAX_TAU = 2 * MIC_SPACING / SPEED_OF_SOUND
CC_LENGTH = 64

# Training config
EPOCHS = 100
LEARNING_RATE = 0.0005
BATCH_SIZE = 32

# Data collection config
STEP_SIZE = 10          # Degrees between positions
TOTAL_RANGE = 130       # Total angle sweep
REPS_PER_SOUND = 2      # Repetitions per sound type per position

# Sound types from Arduino
SOUND_TYPES = [
    ('0', 'Chirp'),
    ('1', '500Hz'),
    ('2', '1000Hz'),
    ('3', '2000Hz'),
    ('4', '4000Hz'),
    ('5', 'Noise'),
    ('6', 'Voice'),
    ('7', 'Click'),
]

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =============================================================================
# CNN MODEL
# =============================================================================

class DoACNN(nn.Module):
    def __init__(self):
        super(DoACNN, self).__init__()
        self.conv1 = nn.Conv1d(6, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(128 * 8, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 2)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        x = x / (torch.norm(x, dim=1, keepdim=True) + 1e-8)
        return x

# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def compute_gcc_phat(sig1, sig2, fs, n_output=64):
    n = len(sig1) + len(sig2)
    SIG1 = np.fft.rfft(sig1, n=n)
    SIG2 = np.fft.rfft(sig2, n=n)
    R = SIG1 * np.conj(SIG2)
    mag = np.abs(R)
    R = R / (mag + 1e-10)
    cc = np.fft.irfft(R, n=n)
    max_shift = int(MAX_TAU * fs) + n_output // 2
    cc = np.concatenate([cc[-max_shift:], cc[:max_shift+1]])
    center = len(cc) // 2
    start = center - n_output // 2
    end = start + n_output
    return cc[start:end].astype(np.float32)

def extract_gcc_features(audio, fs=16000):
    pairs = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
    features = []
    for i, j in pairs:
        cc = compute_gcc_phat(audio[:, i], audio[:, j], fs, CC_LENGTH)
        features.append(cc)
    return np.array(features, dtype=np.float32)

# =============================================================================
# HELPERS
# =============================================================================

def send_goto(ser, angle):
    cmd = f"G{angle:03d}"
    ser.write(cmd.encode())
    while True:
        line = ser.readline().decode().strip()
        if line == "READY":
            return

def send_sound(ser, sound_code):
    cmd = f"S{sound_code}"
    ser.write(cmd.encode())

def angle_loss(pred, target_sin, target_cos):
    target = torch.stack([target_sin, target_cos], dim=1)
    cos_sim = torch.sum(pred * target, dim=1)
    return 1 - cos_sim.mean()

def pred_to_angle(sin_cos):
    angles = np.arctan2(sin_cos[:, 0], sin_cos[:, 1])
    angles = np.degrees(angles)
    return (angles + 360) % 360

# =============================================================================
# DATASET
# =============================================================================

class DoADataset(Dataset):
    def __init__(self, X, y, sound_types):
        self.X = torch.from_numpy(X).float()
        angles_rad = np.radians(y)
        self.y_sin = torch.from_numpy(np.sin(angles_rad)).float()
        self.y_cos = torch.from_numpy(np.cos(angles_rad)).float()
        self.y_deg = torch.from_numpy(y).float()
        self.sound_types = sound_types
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y_sin[idx], self.y_cos[idx], self.y_deg[idx]

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print(" Extended CNN Training with Diverse Sounds")
    print("=" * 60)
    print(f"    Device: {DEVICE}")
    
    # Connect to Arduino
    print("\n[1] Connecting to Arduino...")
    ports = list(serial.tools.list_ports.comports())
    com_port = None
    for p in ports:
        if any(x in p.description.lower() for x in ['arduino', 'ch340', 'usb serial']):
            com_port = p.device
            break
    if com_port is None and ports:
        com_port = ports[0].device
    
    ard = serial.Serial(com_port, BAUD_RATE, timeout=2)
    time.sleep(2)
    ard.read_all()
    print(f"    Connected to {com_port}")
    
    # Connect to ReSpeaker
    print("\n[2] Connecting to ReSpeaker...")
    respeaker_id = None
    for i, dev in enumerate(sd.query_devices()):
        if dev['max_input_channels'] >= 4:
            name = dev['name'].lower()
            if 'respeaker' in name or 'uac1.0' in name:
                respeaker_id = i
                print(f"    Found: {dev['name']}")
                break
    
    if respeaker_id is None:
        print("ERROR: ReSpeaker not found!")
        ard.close()
        return
    
    # Load existing model if available
    print("\n[3] Loading existing model...")
    model = DoACNN().to(DEVICE)
    model_path = Path('measure_1/doa_cnn_finetuned.pth')
    
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print("    Loaded previous fine-tuned model")
    elif Path('measure_1/doa_cnn_model.pth').exists():
        model.load_state_dict(torch.load('measure_1/doa_cnn_model.pth', map_location=DEVICE))
        print("    Loaded pre-trained model")
    else:
        print("    Starting fresh")
    
    # ==========================================================================
    # DATA COLLECTION
    # ==========================================================================
    
    print("\n" + "=" * 60)
    print(" Collecting Diverse Training Data")
    print("=" * 60)
    
    n_positions = TOTAL_RANGE // STEP_SIZE + 1
    n_sounds = len(SOUND_TYPES)
    total_samples = n_positions * n_sounds * REPS_PER_SOUND
    
    print(f"    Positions: {n_positions} (every {STEP_SIZE}°)")
    print(f"    Sound types: {n_sounds}")
    print(f"    Reps per sound: {REPS_PER_SOUND}")
    print(f"    Total samples: {total_samples}")
    print(f"    Estimated time: ~{total_samples * 1.5 / 60:.1f} minutes")
    
    input("\n    Press ENTER to start data collection...")
    
    # Move to start
    start_angle = 45
    send_goto(ard, start_angle)
    
    data_X = []
    data_y = []
    data_sounds = []
    
    try:
        for pos_idx in range(n_positions):
            true_angle = pos_idx * STEP_SIZE
            servo_pos = start_angle + true_angle
            
            if servo_pos > 180:
                print(f"    Reached servo limit at {servo_pos}°")
                break
            
            print(f"\n--- Position {pos_idx+1}/{n_positions}: {true_angle}° (servo {servo_pos}°) ---")
            send_goto(ard, servo_pos)
            
            for sound_code, sound_name in SOUND_TYPES:
                for rep in range(REPS_PER_SOUND):
                    # Record
                    recording = sd.rec(
                        int(RECORDING_DURATION * SAMPLE_RATE),
                        samplerate=SAMPLE_RATE,
                        channels=6,
                        device=respeaker_id,
                        dtype='int16'
                    )
                    time.sleep(0.1)
                    send_sound(ard, sound_code)
                    sd.wait()
                    
                    # Extract features
                    raw_audio = recording[:, 1:5].astype(np.float64)
                    features = extract_gcc_features(raw_audio, SAMPLE_RATE)
                    
                    data_X.append(features)
                    data_y.append(true_angle)
                    data_sounds.append(sound_name)
                    
                print(f"    {sound_name}: {REPS_PER_SOUND} samples", end="  ")
            print()
            
    except KeyboardInterrupt:
        print("\n    Data collection interrupted!")
    
    ard.close()
    
    if len(data_X) < 20:
        print("ERROR: Not enough data!")
        return
    
    X = np.array(data_X)
    y = np.array(data_y)
    sounds = np.array(data_sounds)
    
    print(f"\n    Collected {len(X)} samples total")
    
    # Save data
    with open('measure_1/extended_training_data.pkl', 'wb') as f:
        pickle.dump({'X': X, 'y': y, 'sounds': sounds}, f)
    print("    Saved to extended_training_data.pkl")
    
    # ==========================================================================
    # TRAINING
    # ==========================================================================
    
    print("\n" + "=" * 60)
    print(" Training CNN on Real Data")
    print("=" * 60)
    
    # Split train/val
    n_samples = len(X)
    n_train = int(n_samples * 0.8)
    indices = np.random.permutation(n_samples)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    
    train_dataset = DoADataset(X[train_idx], y[train_idx], sounds[train_idx])
    val_dataset = DoADataset(X[val_idx], y[val_idx], sounds[val_idx])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    print(f"    Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)
    
    train_losses = []
    val_errors = []
    best_val_error = float('inf')
    
    for epoch in range(EPOCHS):
        # Training
        model.train()
        epoch_loss = 0
        for X_batch, y_sin, y_cos, y_deg in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_sin = y_sin.to(DEVICE)
            y_cos = y_cos.to(DEVICE)
            
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = angle_loss(pred, y_sin, y_cos)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        epoch_loss /= len(train_loader)
        train_losses.append(epoch_loss)
        
        # Validation
        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for X_batch, y_sin, y_cos, y_deg in val_loader:
                X_batch = X_batch.to(DEVICE)
                pred = model(X_batch)
                all_preds.append(pred.cpu().numpy())
                all_targets.append(y_deg.numpy())
        
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        pred_angles = pred_to_angle(all_preds)
        
        errors = np.abs(pred_angles - all_targets)
        errors = np.minimum(errors, 360 - errors)
        val_error = np.mean(errors)
        val_errors.append(val_error)
        
        scheduler.step(val_error)
        
        if val_error < best_val_error:
            best_val_error = val_error
            torch.save(model.state_dict(), 'measure_1/doa_cnn_finetuned.pth')
        
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{EPOCHS} | Loss: {epoch_loss:.4f} | Val Error: {val_error:.1f}°")
    
    print(f"\n    Best validation error: {best_val_error:.1f}°")
    print("    Model saved: measure_1/doa_cnn_finetuned.pth")
    
    # ==========================================================================
    # EVALUATION BY SOUND TYPE
    # ==========================================================================
    
    print("\n" + "=" * 60)
    print(" Performance by Sound Type")
    print("=" * 60)
    
    model.eval()
    X_tensor = torch.from_numpy(X).float().to(DEVICE)
    
    with torch.no_grad():
        all_pred = model(X_tensor)
    pred_angles = pred_to_angle(all_pred.cpu().numpy())
    
    sound_errors = {}
    for sound_name in [s[1] for s in SOUND_TYPES]:
        mask = sounds == sound_name
        if np.any(mask):
            errs = np.abs(pred_angles[mask] - y[mask])
            errs = np.minimum(errs, 360 - errs)
            sound_errors[sound_name] = np.mean(errs)
            print(f"    {sound_name:10s}: {np.mean(errs):5.1f}°")
    
    # ==========================================================================
    # PLOTTING
    # ==========================================================================
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Extended CNN Training Results", fontsize=16, fontweight='bold')
    
    # Training curves
    ax1 = axes[0, 0]
    ax1.plot(train_losses, 'b-', label='Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Validation error
    ax2 = axes[0, 1]
    ax2.plot(val_errors, 'g-')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Error (°)')
    ax2.set_title('Validation Error')
    ax2.grid(True, alpha=0.3)
    
    # Error by sound type
    ax3 = axes[1, 0]
    sound_names = list(sound_errors.keys())
    sound_vals = list(sound_errors.values())
    colors = plt.cm.viridis(np.linspace(0, 1, len(sound_names)))
    bars = ax3.bar(sound_names, sound_vals, color=colors)
    ax3.set_ylabel('Mean Error (°)')
    ax3.set_title('Error by Sound Type')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, sound_vals):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.1f}°', ha='center', fontsize=9)
    
    # Tracking accuracy
    ax4 = axes[1, 1]
    ax4.scatter(y, pred_angles, alpha=0.3, s=10)
    ax4.plot([0, max(y)], [0, max(y)], 'r--', linewidth=2, label='Perfect')
    ax4.set_xlabel('True Angle (°)')
    ax4.set_ylabel('Estimated Angle (°)')
    ax4.set_title('Tracking Accuracy (All Sounds)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('measure_1/extended_training_results.png', dpi=150)
    print(f"\n    Plot saved: measure_1/extended_training_results.png")
    plt.show()
    
    print("\nDone!")

if __name__ == "__main__":
    main()
