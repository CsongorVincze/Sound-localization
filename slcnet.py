import torch
import torch.nn as nn

class SLCnet(nn.Module):
    """
    Sound Localization and Classification Network (SLCnet)
    As proposed in "SLoClas: A DATABASE FOR JOINT SOUND LOCALIZATION AND CLASSIFICATION"
    """
    def __init__(self, input_dim=618, num_classes=10):
        super(SLCnet, self).__init__()
        
        # --- 1. General Embedding Extraction Block ---
        # Note: The paper processes features sequentially (Batch, Time, Features).
        # We apply the FC layers across the feature dimension for each time step.
        self.embedding_block = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )
        
        # --- 2. DOAE (Localization) Branch ---
        self.doae_branch = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Linear(1024, 360)  # 360-dim Gaussian likelihood target for MSE loss
        )

        # --- 3. SEC (Classification) Branch ---
        self.sec_branch = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Linear(1024, num_classes),
            nn.Sigmoid()  # matches original training code
        )

    def forward(self, x):
        # x shape: (Batch, Time, 618)
        B, T, F = x.shape
        
        # Flatten Batch and Time to apply FC layers easily
        x_flat = x.view(B * T, F)
        
        # Extract embeddings
        embeddings = self.embedding_block(x_flat) # (B*T, 1024)
        
        # Reshape back to (Batch, 1024, Time) for Max Pooling
        embeddings = embeddings.view(B, T, 1024).permute(0, 2, 1) # (B, 1024, T)
        
        # "A max pooling operation is applied along the time dimension"
        pooled_embeddings, _ = torch.max(embeddings, dim=2) # (B, 1024)
        
        # Pass the single temporal embedding through both branches
        doa_pred = self.doae_branch(pooled_embeddings) # (B, 360)
        sec_pred = self.sec_branch(pooled_embeddings) # (B, 10)
        
        return doa_pred, sec_pred
