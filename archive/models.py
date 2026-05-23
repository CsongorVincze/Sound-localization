import torch
import torch.nn as nn
import torchvision.models as models

# ---------------------------------------------------------
# 1. Tiny CRNN (Lightweight Baseline)
# ---------------------------------------------------------
class TinyCRNN(nn.Module):
    """Extremely lightweight CNN + GRU. Perfect for edge devices."""
    def __init__(self, in_channels=8):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1), 
            nn.BatchNorm2d(16), 
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), 
            nn.BatchNorm2d(32), 
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((1, None)) # Pool frequency, keep time
        )
        self.rnn = nn.GRU(32, 32, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(64, 2) # [sin, cos]

    def forward(self, x):
        # x: (B, C, F, T) -> (B, 32, 1, T)
        x = self.cnn(x).squeeze(2).permute(0, 2, 1) # -> (B, T, 32)
        x, _ = self.rnn(x)
        return self.fc(x[:, -1, :]) # Take last time step

# ---------------------------------------------------------
# 2. Lightweight MobileNetV2 (Efficient CNN)
# ---------------------------------------------------------
class LightweightMobileNet(nn.Module):
    """Uses depthwise separable convolutions to save massive amounts of parameters."""
    def __init__(self, in_channels=8):
        super().__init__()
        # Load an empty MobileNetV2 architecture
        self.mobilenet = models.mobilenet_v2(weights=None)
        # Modify the first layer to accept our 8-channel complex spectrogram
        self.mobilenet.features[0][0] = nn.Conv2d(
            in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False
        )
        # Modify the classifier to output 2 coordinates
        self.mobilenet.classifier[1] = nn.Linear(self.mobilenet.last_channel, 2)

    def forward(self, x):
        return self.mobilenet(x)

# ---------------------------------------------------------
# 3. Transformer-CRNN (Heavy Attention)
# ---------------------------------------------------------
class TransformerCRNN(nn.Module):
    """Uses Self-Attention to find correlations across the entire audio sequence."""
    def __init__(self, in_channels=8):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1), 
            nn.BatchNorm2d(64), 
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), 
            nn.BatchNorm2d(128), 
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((1, None))
        )
        # 4-Layer Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=8, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.fc = nn.Linear(128, 2)

    def forward(self, x):
        x = self.cnn(x).squeeze(2).permute(0, 2, 1) # -> (B, T, 128)
        x = self.transformer(x)
        # Average pooling over the time dimension
        x = x.mean(dim=1) 
        return self.fc(x)

# ---------------------------------------------------------
# 4. ResNet-18 (Robust Baseline)
# ---------------------------------------------------------
class ResNetSSL(nn.Module):
    """Heavy ResNet operating on spatial Spectrograms."""
    def __init__(self, in_channels=8): 
        super().__init__()
        self.resnet = models.resnet18(weights=None)
        self.resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 2)

    def forward(self, x):
        return self.resnet(x)

# ---------------------------------------------------------
# 5. End-to-End Waveform 1D-CNN
# ---------------------------------------------------------
class Waveform1DCNN(nn.Module):
    """Learns directly from raw audio waveforms without STFT."""
    def __init__(self, in_channels=4):
        super().__init__()
        self.cnn = nn.Sequential(
            # Massive kernel to act as a learned STFT filterbank
            nn.Conv1d(in_channels, 64, kernel_size=251, stride=10, padding=125),
            nn.BatchNorm1d(64), 
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(64, 128, kernel_size=11, stride=2, padding=5),
            nn.BatchNorm1d(128), 
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)
        )
        self.fc = nn.Sequential(
            nn.Linear(128, 64), 
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        # x expected shape: (Batch, 4, TimeSamples)
        x = self.cnn(x).squeeze(-1)
        return self.fc(x)

# ---------------------------------------------------------
# Helper function
# ---------------------------------------------------------
def get_model(model_name):
    """Factory function to instantiate the selected architecture."""
    models_dict = {
        'tiny': TinyCRNN(in_channels=8),
        'mobilenet': LightweightMobileNet(in_channels=8),
        'transformer': TransformerCRNN(in_channels=8),
        'resnet': ResNetSSL(in_channels=8),
        'waveform': Waveform1DCNN(in_channels=4) # Expects raw 4-channel audio
    }
    
    if model_name not in models_dict:
        raise ValueError(f"Unknown model architecture: '{model_name}'. Available: {list(models_dict.keys())}")
        
    return models_dict[model_name]
