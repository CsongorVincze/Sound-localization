import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------
# 1. Complex U-Net + Transformer (Frequency Domain)
# ---------------------------------------------------------
class ComplexUNetTransformer(nn.Module):
    """
    Uses the Fourier Transform (STFT) via 8-channel Complex Spectrograms.
    U-Net predicts continuous separation masks.
    Transformer Encoder uses bottleneck features to track DoA.
    """
    def __init__(self, in_channels=8, num_sources=2):
        super().__init__()
        self.num_sources = num_sources
        
        # U-Net Encoder
        self.enc1 = nn.Sequential(nn.Conv2d(in_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU())
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU())
        
        # U-Net Decoder (for separation masks)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU())
        
        # Predicts 1 mask per source
        self.mask_out = nn.Sequential(nn.Conv2d(32, num_sources, 1), nn.Sigmoid())
        
        # Transformer Branch (for DoA)
        self.doa_pool = nn.AdaptiveMaxPool2d((1, None))
        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=8, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.doa_out = nn.Linear(128, num_sources * 2) # [sin, cos] for each source

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        
        # Mask Decoder
        d1 = self.up1(e3)
        if d1.shape != e2.shape: d1 = F.interpolate(d1, size=e2.shape[2:])
        d1 = self.dec1(torch.cat([d1, e2], dim=1))
        
        d2 = self.up2(d1)
        if d2.shape != e1.shape: d2 = F.interpolate(d2, size=e1.shape[2:])
        d2 = self.dec2(torch.cat([d2, e1], dim=1))
        
        masks = self.mask_out(d2) # (B, 2, F, T)
        
        # DoA Branch (from deep features e3)
        doa_feat = self.doa_pool(e3).squeeze(2).permute(0, 2, 1) # (B, T, 128)
        doa_feat = self.transformer(doa_feat)
        doa_feat = doa_feat.mean(dim=1) # Global average over time
        doas = self.doa_out(doa_feat).view(-1, self.num_sources, 2) # (B, 2, 2)
        
        return masks, doas

# ---------------------------------------------------------
# 2. Spatial Conv-TasNet (Time Domain)
# ---------------------------------------------------------
class SpatialConvTasNet(nn.Module):
    """
    Operates on the raw waveform (no STFT) across all 4 mics.
    Uses dilated 1D Convolutions (TCN) to untangle the audio.
    """
    def __init__(self, in_channels=4, num_sources=2):
        super().__init__()
        self.num_sources = num_sources
        
        # Learned Encoder (replaces STFT)
        self.encoder = nn.Conv1d(in_channels, 256, kernel_size=16, stride=8, padding=4)
        
        # Separation TCN (Dilated Convolutions)
        self.tcn = nn.Sequential(
            nn.Conv1d(256, 512, 1),
            nn.BatchNorm1d(512),
            nn.PReLU(),
            nn.Conv1d(512, 512, 3, dilation=1, padding=1, groups=512),
            nn.BatchNorm1d(512),
            nn.PReLU(),
            nn.Conv1d(512, 256, 1)
        )
        
        # Mask Generation
        self.mask_out = nn.Sequential(
            nn.Conv1d(256, num_sources * 256, 1),
            nn.Sigmoid()
        )
        
        # DoA Branch
        self.doa_pool = nn.AdaptiveAvgPool1d(1)
        self.doa_out = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_sources * 2)
        )
        
        # Learned Decoder (reconstructs separated waveforms)
        self.decoder = nn.ConvTranspose1d(256, 1, kernel_size=16, stride=8, padding=4)

    def forward(self, x):
        # x is (B, 4, T)
        enc = self.encoder(x) # (B, 256, L)
        
        tcn_out = self.tcn(enc) # (B, 256, L)
        
        # Masks
        masks = self.mask_out(tcn_out).view(x.size(0), self.num_sources, 256, -1) # (B, 2, 256, L)
        
        # Apply masks to encoder features
        enc_expanded = enc.unsqueeze(1) # (B, 1, 256, L)
        masked_features = enc_expanded * masks # (B, 2, 256, L)
        
        # Decode to waveforms
        B, S, C, L = masked_features.shape
        separated_audio = self.decoder(masked_features.view(B * S, C, L))
        separated_audio = separated_audio.view(B, S, -1) # (B, 2, Time)
        
        # Predict DoA
        doa_feat = self.doa_pool(tcn_out).squeeze(-1) # (B, 256)
        doas = self.doa_out(doa_feat).view(B, self.num_sources, 2) # (B, 2, 2)
        
        return separated_audio, doas

# ---------------------------------------------------------
# 3. FaSNet (Filter-and-Sum Network)
# ---------------------------------------------------------
class FaSNet(nn.Module):
    """
    Learns explicit spatial beamforming filters to steer the 4 microphones.
    """
    def __init__(self, in_channels=4, num_sources=2):
        super().__init__()
        self.num_sources = num_sources
        self.in_channels = in_channels
        
        # Encoder
        self.encoder = nn.Conv1d(1, 64, kernel_size=16, stride=8, padding=4)
        
        # Spatial feature extractor (processes all channels)
        self.spatial_cnn = nn.Sequential(
            nn.Conv1d(64 * in_channels, 128, 1),
            nn.PReLU(),
            nn.Conv1d(128, 128, 3, padding=1),
            nn.PReLU()
        )
        
        # Beamforming Filter Predictor
        self.filter_out = nn.Conv1d(128, num_sources * in_channels * 64, 1)
        
        # DoA Predictor
        self.doa_pool = nn.AdaptiveAvgPool1d(1)
        self.doa_out = nn.Linear(128, num_sources * 2)
        
        self.decoder = nn.ConvTranspose1d(64, 1, kernel_size=16, stride=8, padding=4)

    def forward(self, x):
        B, C, T = x.shape
        
        # Encode each channel independently
        x_flat = x.view(B * C, 1, T)
        enc = self.encoder(x_flat)
        _, F_dim, L = enc.shape
        enc = enc.view(B, C * F_dim, L)
        
        # Extract spatial features
        spatial = self.spatial_cnn(enc) # (B, 128, L)
        
        # Predict spatial filters for each source and channel
        filters = self.filter_out(spatial) # (B, num_sources * C * F_dim, L)
        filters = filters.view(B, self.num_sources, C, F_dim, L)
        
        # Apply filter-and-sum beamforming
        # enc is reshaped to (B, 1, C, F_dim, L)
        enc_expanded = enc.view(B, 1, C, F_dim, L)
        filtered_enc = (enc_expanded * filters).sum(dim=2) # Sum across microphones! (B, 2, F_dim, L)
        
        # Decode
        separated_audio = self.decoder(filtered_enc.view(B * self.num_sources, F_dim, L))
        separated_audio = separated_audio.view(B, self.num_sources, -1)
        
        # DoA
        doa_feat = self.doa_pool(spatial).squeeze(-1)
        doas = self.doa_out(doa_feat).view(B, self.num_sources, 2)
        
        return separated_audio, doas

# ---------------------------------------------------------
# Factory Function
# ---------------------------------------------------------
def get_sep_model(model_name):
    models = {
        'unet': ComplexUNetTransformer(),
        'tasnet': SpatialConvTasNet(),
        'fasnet': FaSNet()
    }
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}")
    return models[model_name]
