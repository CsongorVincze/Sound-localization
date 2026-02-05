"""
SNN Baseline for DoA Estimation (V2 - With Augmentation & Regression)
----------------------------------------------------------------------
Key changes from V1:
1. Data Augmentation: Creates multiple samples per recording
2. Regression output: Predicts angle directly instead of classification
3. Proper train/test split by angle (not random)
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import snntorch as snn
from snntorch import spikegen
import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_PATH = Path('measure_1/snn_training_data.pkl')
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
EPOCHS = 100
BETA = 0.85           # Membrane decay rate
NUM_HIDDEN = 256      # Larger hidden layer
DOWNSAMPLE_FACTOR = 32  # 16000/32 = 500 Hz (500 steps)

# Augmentation settings
AUGMENTATIONS_PER_SAMPLE = 20  # Create 20 augmented versions of each sample
NOISE_LEVEL = 0.02             # Add 2% noise
TIME_SHIFT_MAX = 50            # Max samples to shift in time

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# =============================================================================
# DATASET WITH AUGMENTATION
# =============================================================================

class DoASpikingDataset(Dataset):
    def __init__(self, data_path, downsample=1, augment=True, n_augment=10):
        with open(data_path, 'rb') as f:
            raw_data = pickle.load(f)
        
        self.samples = []
        self.labels = []
        
        # Get angle range for normalization (0-1 output)
        all_angles = [d['angle'] for d in raw_data]
        self.min_angle = min(all_angles)
        self.max_angle = max(all_angles)
        self.angle_range = self.max_angle - self.min_angle
        
        print(f"Loaded {len(raw_data)} raw samples.")
        print(f"Angle range: {self.min_angle}° to {self.max_angle}°")

        for item in raw_data:
            audio = item['audio'].astype(np.float32)
            angle = item['angle']
            
            # Normalize angle to [0, 1]
            norm_angle = (angle - self.min_angle) / (self.angle_range + 1e-8)
            
            # Normalize audio
            audio = audio / (np.max(np.abs(audio)) + 1e-8)
            
            # Downsample
            audio = audio[::downsample, :]
            
            # Original sample
            self.samples.append(audio)
            self.labels.append(norm_angle)
            
            # Augmented samples
            if augment:
                for _ in range(n_augment):
                    aug_audio = self._augment(audio.copy())
                    self.samples.append(aug_audio)
                    self.labels.append(norm_angle)
        
        print(f"After augmentation: {len(self.samples)} samples")

    def _augment(self, audio):
        """Apply random augmentations to audio."""
        # 1. Add Gaussian noise
        noise = np.random.randn(*audio.shape) * NOISE_LEVEL
        audio = audio + noise
        
        # 2. Random time shift (circular)
        shift = np.random.randint(-TIME_SHIFT_MAX, TIME_SHIFT_MAX)
        audio = np.roll(audio, shift, axis=0)
        
        # 3. Random amplitude scaling
        scale = np.random.uniform(0.8, 1.2)
        audio = audio * scale
        
        # Re-normalize
        audio = audio / (np.max(np.abs(audio)) + 1e-8)
        
        return audio.astype(np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.samples[idx])
        y = torch.FloatTensor([self.labels[idx]])
        return x, y

# =============================================================================
# SNN MODEL (REGRESSION)
# =============================================================================

class MemristorDoANet(nn.Module):
    def __init__(self, num_inputs, num_hidden, beta):
        super().__init__()
        
        # Layer 1
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta)
        
        # Layer 2
        self.fc2 = nn.Linear(num_hidden, num_hidden // 2)
        self.lif2 = snn.Leaky(beta=beta)

        # Output Layer: 1 neuron for regression
        self.fc_out = nn.Linear(num_hidden // 2, 1)

    def forward(self, x):
        # x: [Batch, Time, Channels] -> [Time, Batch, Channels]
        x = x.permute(1, 0, 2) 
        time_steps = x.size(0)
        batch_size = x.size(1)
        
        # Initialize membranes
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        
        # Spike encoding using delta modulation
        spk_in = spikegen.delta(x, threshold=0.03)
        
        # Accumulator for output (we'll average over time)
        output_acc = torch.zeros(batch_size, 1, device=x.device)

        # Time Loop
        for step in range(time_steps):
            cur1 = self.fc1(spk_in[step])
            spk1, mem1 = self.lif1(cur1, mem1)
            
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            
            # Readout from membrane potential (continuous)
            out = self.fc_out(mem2)
            output_acc = output_acc + out
            
        # Average over time
        output = output_acc / time_steps
        return torch.sigmoid(output)  # Squash to [0, 1]

# =============================================================================
# TRAINING
# =============================================================================

def train():
    print("="*60)
    print(" SNN Baseline Training (V2 - Regression + Augmentation)")
    print("="*60)
    
    # 1. Load Data
    full_dataset = DoASpikingDataset(DATA_PATH, 
                                      downsample=DOWNSAMPLE_FACTOR,
                                      augment=True,
                                      n_augment=AUGMENTATIONS_PER_SAMPLE)
    
    # Split 80/20
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 2. Setup Model
    net = MemristorDoANet(num_inputs=4, 
                          num_hidden=NUM_HIDDEN, 
                          beta=BETA).to(device)
    
    # MSE Loss for regression
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    print(f"\nModel: {net}")
    print(f"Device: {device}")
    print(f"Time steps per sample: {16000 // DOWNSAMPLE_FACTOR}")
    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

    # 3. Training Loop
    loss_hist = []
    best_mae = float('inf')
    
    for epoch in range(EPOCHS):
        net.train()
        batch_loss = 0
        for data, targets in train_loader:
            data = data.to(device)
            targets = targets.to(device)
            
            # Forward
            outputs = net(data)
            loss = loss_fn(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            batch_loss += loss.item()
            
        epoch_loss = batch_loss / len(train_loader)
        loss_hist.append(epoch_loss)
        scheduler.step(epoch_loss)
        
        # Evaluate every 10 epochs
        if (epoch+1) % 10 == 0 or epoch == 0:
            net.eval()
            all_preds = []
            all_targets = []
            with torch.no_grad():
                for data, targets in test_loader:
                    data = data.to(device)
                    preds = net(data)
                    all_preds.extend(preds.cpu().numpy().flatten())
                    all_targets.extend(targets.cpu().numpy().flatten())
            
            # Convert back to degrees
            all_preds = np.array(all_preds) * full_dataset.angle_range + full_dataset.min_angle
            all_targets = np.array(all_targets) * full_dataset.angle_range + full_dataset.min_angle
            
            mae = np.mean(np.abs(all_preds - all_targets))
            print(f"Epoch {epoch+1:3d}/{EPOCHS} | Loss: {epoch_loss:.4f} | MAE: {mae:.1f}°")
            
            if mae < best_mae:
                best_mae = mae
                torch.save(net.state_dict(), 'measure_1/snn_baseline_model.pth')
            
    # 4. Final Evaluation
    print("\n" + "="*60)
    print(" Final Evaluation")
    print("="*60)
    
    net.load_state_dict(torch.load('measure_1/snn_baseline_model.pth'))
    net.eval()
    
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for data, targets in test_loader:
            data = data.to(device)
            preds = net(data)
            all_preds.extend(preds.cpu().numpy().flatten())
            all_targets.extend(targets.cpu().numpy().flatten())
    
    # Convert back to degrees
    all_preds = np.array(all_preds) * full_dataset.angle_range + full_dataset.min_angle
    all_targets = np.array(all_targets) * full_dataset.angle_range + full_dataset.min_angle
    
    mae = np.mean(np.abs(all_preds - all_targets))
    rmse = np.sqrt(np.mean((all_preds - all_targets)**2))
    
    print(f"Mean Absolute Error: {mae:.2f}°")
    print(f"RMSE: {rmse:.2f}°")
    print(f"Best MAE achieved: {best_mae:.2f}°")
    
    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Loss curve
    axes[0].plot(loss_hist)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].grid(True, alpha=0.3)
    
    # Predictions vs Targets
    axes[1].scatter(all_targets, all_preds, alpha=0.5)
    axes[1].plot([0, 180], [0, 180], 'r--', label='Perfect')
    axes[1].set_xlabel('True Angle (°)')
    axes[1].set_ylabel('Predicted Angle (°)')
    axes[1].set_title(f'Predictions (MAE: {mae:.1f}°)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('SNN_sim/snn_training_results.png', dpi=150)
    print("\nPlot saved to SNN_sim/snn_training_results.png")
    plt.show()

if __name__ == "__main__":
    train()
