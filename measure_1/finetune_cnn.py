"""
CNN Real-Time Fine-Tuning for DoA
Collects real data from hardware and fine-tunes the pre-trained CNN.
"""
import serial
import serial.tools.list_ports
import time
import sounddevice as sd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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

# Fine-tuning config
FINETUNE_EPOCHS = 50
FINETUNE_LR = 0.0001  # Lower learning rate for fine-tuning
BATCH_SIZE = 16

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
# HELPER FUNCTIONS
# =============================================================================

def send_goto(ser, angle):
    cmd = f"G{angle:03d}"
    ser.write(cmd.encode())
    while True:
        line = ser.readline().decode().strip()
        if line == "READY":
            return

def send_play(ser):
    ser.write(b'P')

def angle_loss(pred, target_sin, target_cos):
    target = torch.stack([target_sin, target_cos], dim=1)
    cos_sim = torch.sum(pred * target, dim=1)
    return 1 - cos_sim.mean()

def pred_to_angle(sin_cos):
    angles = np.arctan2(sin_cos[:, 0], sin_cos[:, 1])
    angles = np.degrees(angles)
    return (angles + 360) % 360

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print(" CNN Real-Time Fine-Tuning")
    print("=" * 60)
    
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
    
    if not com_port:
        print("ERROR: No COM ports!")
        return
    
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
    
    # Load pre-trained model
    print("\n[3] Loading pre-trained model...")
    model = DoACNN().to(DEVICE)
    model_path = Path('doa_cnn_model.pth')
    
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print("    Loaded pre-trained weights")
    else:
        print("    WARNING: No pre-trained model, starting fresh")
    
    # ==========================================================================
    # COLLECT REAL TRAINING DATA
    # ==========================================================================
    
    print("\n" + "=" * 60)
    print(" Collecting Real Training Data")
    print("=" * 60)
    
    # Move to start position
    start_angle = 45
    send_goto(ard, start_angle)
    
    # Collect data across different positions
    step_size = 5
    total_steps = 27  # 135 degrees
    reps_per_step = 3  # Multiple samples per position
    
    real_data_X = []
    real_data_y = []
    
    print(f"    Steps: {total_steps}, Reps per step: {reps_per_step}")
    print(f"    Total samples: {total_steps * reps_per_step}")
    
    try:
        current_angle = 0
        servo_pos = start_angle
        
        for step in range(total_steps):
            print(f"\n--- Position {step+1}/{total_steps}: {current_angle}° ---")
            
            # Move servo
            send_goto(ard, servo_pos)
            
            for rep in range(reps_per_step):
                # Record
                recording = sd.rec(
                    int(RECORDING_DURATION * SAMPLE_RATE),
                    samplerate=SAMPLE_RATE,
                    channels=6,
                    device=respeaker_id,
                    dtype='int16'
                )
                time.sleep(0.1)
                send_play(ard)
                sd.wait()
                
                # Extract features
                raw_audio = recording[:, 1:5].astype(np.float64)
                features = extract_gcc_features(raw_audio, SAMPLE_RATE)
                
                real_data_X.append(features)
                real_data_y.append(current_angle)
                
                print(f"    Rep {rep+1}: collected", end=" ")
            print()
            
            current_angle += step_size
            servo_pos += step_size
            
    except KeyboardInterrupt:
        print("\n    Data collection interrupted!")
    
    ard.close()
    
    if len(real_data_X) < 10:
        print("ERROR: Not enough data collected!")
        return
    
    X_real = np.array(real_data_X)
    y_real = np.array(real_data_y)
    
    print(f"\n    Collected {len(X_real)} real samples")
    
    # Save real data
    with open('real_training_data.pkl', 'wb') as f:
        pickle.dump({'X': X_real, 'y': y_real}, f)
    print("    Saved to real_training_data.pkl")
    
    # ==========================================================================
    # FINE-TUNE THE MODEL
    # ==========================================================================
    
    print("\n" + "=" * 60)
    print(" Fine-Tuning on Real Data")
    print("=" * 60)
    
    # Prepare data
    angles_rad = np.radians(y_real)
    y_sin = torch.from_numpy(np.sin(angles_rad)).float().to(DEVICE)
    y_cos = torch.from_numpy(np.cos(angles_rad)).float().to(DEVICE)
    X_tensor = torch.from_numpy(X_real).float().to(DEVICE)
    
    # Split train/val
    n_samples = len(X_tensor)
    n_train = int(n_samples * 0.8)
    indices = np.random.permutation(n_samples)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    
    # Optimizer with lower learning rate for fine-tuning
    optimizer = optim.Adam(model.parameters(), lr=FINETUNE_LR)
    
    # Training loop
    train_losses = []
    val_errors = []
    
    print(f"    Training samples: {len(train_idx)}")
    print(f"    Validation samples: {len(val_idx)}")
    print(f"    Epochs: {FINETUNE_EPOCHS}")
    print(f"    Learning rate: {FINETUNE_LR}")
    
    best_val_error = float('inf')
    
    for epoch in range(FINETUNE_EPOCHS):
        # Training
        model.train()
        
        # Shuffle training data
        np.random.shuffle(train_idx)
        
        epoch_loss = 0
        for i in range(0, len(train_idx), BATCH_SIZE):
            batch_idx = train_idx[i:i+BATCH_SIZE]
            
            X_batch = X_tensor[batch_idx]
            y_sin_batch = y_sin[batch_idx]
            y_cos_batch = y_cos[batch_idx]
            
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = angle_loss(pred, y_sin_batch, y_cos_batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        epoch_loss /= (len(train_idx) / BATCH_SIZE)
        train_losses.append(epoch_loss)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_tensor[val_idx])
            val_angles = pred_to_angle(val_pred.cpu().numpy())
            val_true = y_real[val_idx]
            
            errors = np.abs(val_angles - val_true)
            errors = np.minimum(errors, 360 - errors)
            val_error = np.mean(errors)
            val_errors.append(val_error)
        
        if val_error < best_val_error:
            best_val_error = val_error
            torch.save(model.state_dict(), 'doa_cnn_finetuned.pth')
        
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{FINETUNE_EPOCHS} | Loss: {epoch_loss:.4f} | Val Error: {val_error:.1f}°")
    
    print(f"\n    Best validation error: {best_val_error:.1f}°")
    print("    Model saved: doa_cnn_finetuned.pth")
    
    # ==========================================================================
    # COMPARE BEFORE/AFTER
    # ==========================================================================
    
    print("\n" + "=" * 60)
    print(" Comparing Before vs After Fine-Tuning")
    print("=" * 60)
    
    # Load original model
    original_model = DoACNN().to(DEVICE)
    if Path('doa_cnn_model.pth').exists():
        original_model.load_state_dict(torch.load('doa_cnn_model.pth', map_location=DEVICE))
    original_model.eval()
    
    # Load fine-tuned model
    finetuned_model = DoACNN().to(DEVICE)
    finetuned_model.load_state_dict(torch.load('doa_cnn_finetuned.pth', map_location=DEVICE))
    finetuned_model.eval()
    
    with torch.no_grad():
        # Original predictions
        orig_pred = original_model(X_tensor)
        orig_angles = pred_to_angle(orig_pred.cpu().numpy())
        orig_errors = np.abs(orig_angles - y_real)
        orig_errors = np.minimum(orig_errors, 360 - orig_errors)
        
        # Fine-tuned predictions
        ft_pred = finetuned_model(X_tensor)
        ft_angles = pred_to_angle(ft_pred.cpu().numpy())
        ft_errors = np.abs(ft_angles - y_real)
        ft_errors = np.minimum(ft_errors, 360 - ft_errors)
    
    print(f"    Original CNN Mean Error: {np.mean(orig_errors):.1f}°")
    print(f"    Fine-tuned CNN Mean Error: {np.mean(ft_errors):.1f}°")
    print(f"    Improvement: {np.mean(orig_errors) - np.mean(ft_errors):.1f}°")
    
    # ==========================================================================
    # PLOTTING
    # ==========================================================================
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("CNN Fine-Tuning Results", fontsize=16, fontweight='bold')
    
    # Plot 1: Training curves
    ax1 = axes[0, 0]
    ax1.plot(train_losses, 'b-', label='Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Fine-Tuning Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Validation error
    ax2 = axes[0, 1]
    ax2.plot(val_errors, 'g-')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Error (°)')
    ax2.set_title('Validation Error During Fine-Tuning')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Before/After comparison
    ax3 = axes[1, 0]
    methods = ['Original CNN', 'Fine-tuned CNN']
    means = [np.mean(orig_errors), np.mean(ft_errors)]
    colors = ['#ff6b6b', '#4ecdc4']
    bars = ax3.bar(methods, means, color=colors, alpha=0.8)
    for bar, val in zip(bars, means):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}°', ha='center', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Mean Error (°)')
    ax3.set_title('Before vs After Fine-Tuning')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Tracking accuracy comparison
    ax4 = axes[1, 1]
    ax4.scatter(y_real, orig_angles, alpha=0.5, label='Original', color='#ff6b6b', s=30)
    ax4.scatter(y_real, ft_angles, alpha=0.5, label='Fine-tuned', color='#4ecdc4', s=30)
    ax4.plot([0, max(y_real)], [0, max(y_real)], 'k--', label='Perfect')
    ax4.set_xlabel('True Angle (°)')
    ax4.set_ylabel('Estimated Angle (°)')
    ax4.set_title('Tracking Accuracy')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('cnn_finetuning_results.png', dpi=150)
    print(f"\n    Plot saved: cnn_finetuning_results.png")
    plt.show()
    
    print("\nDone! Use 'doa_cnn_finetuned.pth' as your model.")

if __name__ == "__main__":
    main()
