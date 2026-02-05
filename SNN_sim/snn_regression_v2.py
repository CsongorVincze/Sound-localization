"""
SNN Regression Training (Extended Dataset)
----------------------------------------------------------------------
This model uses the high-resolution dataset (150 samples) collected 
with 3-degree steps and 3 repetitions.

Pipeline:
1. Load 'measure_1/snn_training_data.pkl'
2. Augment data (Noise, Shift, Scale)
3. Train Regression SNN (Predict Angle 0-180°)
4. Evaluate with MAE and RMSE
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
OUTPUT_MODEL_PATH = 'measure_1/snn_regression_model_v2.pth'
OUTPUT_PLOT_PATH = 'SNN_sim/snn_regression_v2_results.png'

BATCH_SIZE = 16
LEARNING_RATE = 1e-3
EPOCHS = 100
BETA = 0.9           # Membrane decay rate
NUM_HIDDEN = 256      # Size of hidden layer
DOWNSAMPLE_FACTOR = 32  # 16000/32 = 500 Hz

# Augmentation Settings
AUGMENTATIONS_PER_SAMPLE = 10   # 150 * 10 = 1500 augmented samples
NOISE_LEVEL = 0.03              # Slightly more noise than before
TIME_SHIFT_MAX = 100            # Allow larger time shifts

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# =============================================================================
# DATASET
# =============================================================================

class DoARegressionDataset(Dataset):
    def __init__(self, data_path, downsample=1, augment=True, n_augment=10):
        with open(data_path, 'rb') as f:
            raw_data = pickle.load(f)
        
        self.samples = []
        self.labels = []
        
        # Determine Range
        all_angles = [d['angle'] for d in raw_data]
        self.min_angle = min(all_angles)
        self.max_angle = max(all_angles)
        self.angle_range = self.max_angle - self.min_angle
        
        print(f"Loaded {len(raw_data)} raw samples.")
        print(f"Angle range: {self.min_angle}° to {self.max_angle}° (Span: {self.angle_range}°)")

        for item in raw_data:
            audio = item['audio'].astype(np.float32)
            angle = item['angle']
            
            # Normalize Angle [0, 1]
            norm_angle = (angle - self.min_angle) / (self.angle_range + 1e-8)
            
            # Normalize Audio [-1, 1]
            audio = audio / (np.max(np.abs(audio)) + 1e-8)
            
            # Downsample
            audio = audio[::downsample, :]
            
            # Add Original
            self.samples.append(audio)
            self.labels.append(norm_angle)
            
            # Add Augmented
            if augment:
                for _ in range(n_augment):
                    aug_audio = self._augment(audio.copy())
                    self.samples.append(aug_audio)
                    self.labels.append(norm_angle)
        
        print(f"Total dataset size after augmentation: {len(self.samples)}")

    def _augment(self, audio):
        # 1. Noise
        noise = np.random.randn(*audio.shape) * NOISE_LEVEL
        audio = audio + noise
        
        # 2. Shift (Circular)
        shift = np.random.randint(-TIME_SHIFT_MAX, TIME_SHIFT_MAX)
        audio = np.roll(audio, shift, axis=0)
        
        # 3. Amplitude Scale
        scale = np.random.uniform(0.7, 1.3)
        audio = audio * scale
        
        # Normalize again
        max_val = np.max(np.abs(audio)) + 1e-8
        return (audio / max_val).astype(np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.samples[idx])
        y = torch.FloatTensor([self.labels[idx]])
        return x, y

# =============================================================================
# MODEL
# =============================================================================

class SNNRegressionNet(nn.Module):
    def __init__(self, num_inputs, num_hidden, beta):
        super().__init__()
        
        # Input -> Hidden
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta)
        
        # Hidden -> Hidden 2
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.lif2 = snn.Leaky(beta=beta)

        # Hidden -> Output (Membrane Potential Readout)
        self.fc_out = nn.Linear(num_hidden, 1)

    def forward(self, x):
        # x: [Batch, Time, Channels] -> [Time, Batch, Channels]
        x = x.permute(1, 0, 2)
        time_steps = x.size(0)
        batch_size = x.size(1)
        
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        
        # Delta Encoding
        spk_in = spikegen.delta(x, threshold=0.04)
        
        output_acc = torch.zeros(batch_size, 1, device=x.device)

        for step in range(time_steps):
            cur1 = self.fc1(spk_in[step])
            spk1, mem1 = self.lif1(cur1, mem1)
            
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            
            # Continuous regression output from membrane potential
            out = self.fc_out(mem2)
            output_acc = output_acc + out

        # Mean over time
        output = output_acc / time_steps
        return torch.sigmoid(output)

# =============================================================================
# TRAINING
# =============================================================================

def train_and_evaluate():
    print("="*60)
    print(" SNN Regression Training V2")
    print("="*60)
    
    # 1. Dataset
    dataset = DoARegressionDataset(DATA_PATH, 
                                   downsample=DOWNSAMPLE_FACTOR,
                                   augment=True,
                                   n_augment=AUGMENTATIONS_PER_SAMPLE)
    
    # Split
    total_len = len(dataset)
    train_len = int(0.85 * total_len)
    test_len = total_len - train_len
    
    train_ds, test_ds = torch.utils.data.random_split(dataset, [train_len, test_len])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Train samples: {len(train_ds)}")
    print(f"Test samples:  {len(test_ds)}")

    # 2. Model
    net = SNNRegressionNet(num_inputs=4, num_hidden=NUM_HIDDEN, beta=BETA).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)

    print(f"Device: {device}")

    # 3. Loop
    loss_history = []
    best_mae = float('inf')

    for epoch in range(EPOCHS):
        net.train()
        batch_losses = []
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            preds = net(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            
            batch_losses.append(loss.item())
        
        epoch_loss = np.mean(batch_losses)
        loss_history.append(epoch_loss)
        scheduler.step(epoch_loss)
        
        # Validation every 10 epochs
        if (epoch+1) % 10 == 0 or epoch == 0:
            net.eval()
            val_ae = []
            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.to(device), y.to(device)
                    preds = net(x)
                    
                    # Convert to degrees
                    pred_deg = preds.cpu().numpy() * dataset.angle_range + dataset.min_angle
                    true_deg = y.cpu().numpy() * dataset.angle_range + dataset.min_angle
                    
                    val_ae.extend(np.abs(pred_deg - true_deg).flatten())
            
            mae = np.mean(val_ae)
            print(f"Epoch {epoch+1:3d}/{EPOCHS} | Loss: {epoch_loss:.5f} | Val MAE: {mae:.2f}°")
            
            if mae < best_mae:
                best_mae = mae
                torch.save(net.state_dict(), OUTPUT_MODEL_PATH)

    # 4. Final Results
    print("\n" + "="*60)
    print(" Final Evaluation (Best Model)")
    print("="*60)
    
    net.load_state_dict(torch.load(OUTPUT_MODEL_PATH))
    net.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            preds = net(x)
            
            p_deg = preds.cpu().numpy().flatten() * dataset.angle_range + dataset.min_angle
            t_deg = y.cpu().numpy().flatten() * dataset.angle_range + dataset.min_angle
            
            all_preds.extend(p_deg)
            all_targets.extend(t_deg)
            
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    final_mae = np.mean(np.abs(all_preds - all_targets))
    final_rmse = np.sqrt(np.mean((all_preds - all_targets)**2))
    
    print(f"Final MAE:  {final_mae:.2f}°")
    print(f"Final RMSE: {final_rmse:.2f}°")
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Loss
    axes[0].plot(loss_history)
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].grid(True, alpha=0.3)
    
    # Scatter
    axes[1].scatter(all_targets, all_preds, alpha=0.4, s=10)
    axes[1].plot([dataset.min_angle, dataset.max_angle], 
                 [dataset.min_angle, dataset.max_angle], 'r--', label='Ideal')
    axes[1].set_title(f"Predicted vs True Angle (MAE: {final_mae:.1f}°)")
    axes[1].set_xlabel("True Angle (°)")
    axes[1].set_ylabel("Predicted Angle (°)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_PATH)
    print(f"Plot saved to {OUTPUT_PLOT_PATH}")
    plt.show()

if __name__ == "__main__":
    train_and_evaluate()
