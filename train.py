"""
Train SLCnet on preprocessed SLoClas features.

Usage:
    python train.py --feat_dir ./preprocessed --out_dir . --wts0 0.99

Matches paper: Adam lr=0.001, batch=32, 50 epochs, lambda=0.99 (MSE) + 0.01 (CE).
Saves best model (by validation DoA ACC within 5 deg) to out_dir/best_slcnet_baseline.pth.
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path

from slcnet import SLCnet


class SLCDataset(Dataset):
    def __init__(self, features, doa_targets, class_labels):
        # Keep as numpy; torch.from_numpy shares memory (no copy)
        self.features    = torch.from_numpy(features)
        self.doa_targets = torch.from_numpy(doa_targets)
        self.class_labels = torch.from_numpy(class_labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.doa_targets[idx], self.class_labels[idx]


def evaluate(model, loader, device):
    model.eval()
    n_total = 0
    doa_mae_sum = 0.0
    doa_correct = 0
    sec_correct = 0

    with torch.no_grad():
        for feat, doa_gt, cls_gt in loader:
            feat   = feat.to(device)
            doa_gt = doa_gt.to(device)
            cls_gt = cls_gt.to(device)

            doa_pred, sec_pred = model(feat)

            # Argmax over 360-dim output gives 0-indexed bin; +1 -> degrees 1..360
            pred_deg = doa_pred.argmax(dim=1) + 1
            true_deg = doa_gt.argmax(dim=1) + 1

            # Circular angular error
            err = (pred_deg - true_deg).abs().float()
            err = torch.minimum(err, 360 - err)

            doa_mae_sum += err.sum().item()
            doa_correct += (err <= 5).sum().item()
            sec_correct += (sec_pred.argmax(dim=1) == cls_gt).sum().item()
            n_total += feat.shape[0]

    return {
        'mae':     doa_mae_sum / n_total,
        'doa_acc': doa_correct / n_total * 100.0,
        'sec_acc': sec_correct / n_total * 100.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feat_dir',    default='./preprocessed')
    parser.add_argument('--out_dir',     default='.')
    parser.add_argument('--wts0',        type=float, default=0.99,
                        help='DoA MSE loss weight; paper reports wts0=0.99')
    parser.add_argument('--epochs',      type=int,   default=50)
    parser.add_argument('--batch',       type=int,   default=32)
    parser.add_argument('--lr',          type=float, default=1e-3)
    parser.add_argument('--num_classes', type=int,   default=10,
                        help='Number of sound classes (10 for SLoClas, 50 for ESC-50)')
    parser.add_argument('--seed',        type=int,   default=7)
    parser.add_argument('--num_workers', type=int,   default=4)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    feat_dir = Path(args.feat_dir)
    out_dir  = Path(args.out_dir)
    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("Loading features ...")
    features     = np.load(feat_dir / 'features.npy')
    doa_targets  = np.load(feat_dir / 'doa_targets.npy')
    class_labels = np.load(feat_dir / 'class_labels.npy').astype(np.int64)
    print(f"  {features.shape[0]} samples, shape {features.shape}")

    dataset = SLCDataset(features, doa_targets, class_labels)
    n_train = int(0.7 * len(dataset))
    n_test  = len(dataset) - n_train
    train_set, test_set = random_split(
        dataset, [n_train, n_test],
        generator=torch.Generator().manual_seed(args.seed)
    )
    print(f"  Train: {n_train}  |  Test: {n_test}")

    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True,
                              num_workers=args.num_workers, pin_memory=device.type == 'cuda')
    test_loader  = DataLoader(test_set,  batch_size=args.batch, shuffle=False,
                              num_workers=args.num_workers, pin_memory=device.type == 'cuda')

    model     = SLCnet(input_dim=618, num_classes=args.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    mse_loss  = nn.MSELoss()
    ce_loss   = nn.CrossEntropyLoss()

    wts0 = args.wts0
    wts1 = 1.0 - wts0
    best_doa_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for feat, doa_gt, cls_gt in train_loader:
            feat   = feat.to(device)
            doa_gt = doa_gt.to(device)
            cls_gt = cls_gt.to(device)

            doa_pred, sec_pred = model(feat)

            # MSE on 360-dim Gaussian target (DOAE branch, no activation)
            # CE on sigmoid outputs (SEC branch) - matches original training code
            loss = wts0 * mse_loss(doa_pred, doa_gt) + wts1 * ce_loss(sec_pred, cls_gt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * feat.shape[0]

        metrics  = evaluate(model, test_loader, device)
        avg_loss = total_loss / n_train

        print(
            f"Epoch {epoch:3d}/{args.epochs}  loss={avg_loss:.5f}  "
            f"DoA MAE={metrics['mae']:.2f}deg  ACC={metrics['doa_acc']:.2f}%  "
            f"SEC ACC={metrics['sec_acc']:.2f}%"
        )

        if metrics['doa_acc'] > best_doa_acc:
            best_doa_acc = metrics['doa_acc']
            save_path = out_dir / 'best_slcnet_baseline.pth'
            torch.save(model.state_dict(), save_path)
            print(f"  -> best model saved  (DoA ACC={best_doa_acc:.2f}%)")

    print(f"\nDone. Best DoA ACC: {best_doa_acc:.2f}%")


if __name__ == '__main__':
    main()
