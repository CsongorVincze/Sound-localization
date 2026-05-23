"""
Train Res15 on Google Speech Commands v2.

Usage:
    python train.py --data_dir /path/to/speech_commands_v0.02 --out_dir .

Recommended remote-server run:
    python train.py --data_dir /data/speech_commands_v0.02 --out_dir . --num_workers 8
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import CLASS_NAMES, NUM_CLASSES, SpeechCommandsDataset
from model import Res15


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for spec, labels in loader:
            spec, labels = spec.to(device), labels.to(device)
            correct += (model(spec).argmax(1) == labels).sum().item()
            total   += labels.size(0)
    return correct / total * 100.0


def evaluate_per_class(model, loader, device):
    model.eval()
    per_class = {i: [0, 0] for i in range(NUM_CLASSES)}  # {label: [correct, total]}
    with torch.no_grad():
        for spec, labels in loader:
            spec, labels = spec.to(device), labels.to(device)
            preds = model(spec).argmax(1)
            for pred, gt in zip(preds, labels):
                per_class[gt.item()][1] += 1
                if pred == gt:
                    per_class[gt.item()][0] += 1
    return {CLASS_NAMES[i]: (v[0] / max(1, v[1]) * 100.0) for i, v in per_class.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir',     required=True,  help='speech_commands_v0.02 root')
    p.add_argument('--out_dir',      default='.',    help='Where to save best model')
    p.add_argument('--epochs',       type=int,   default=50)
    p.add_argument('--batch',        type=int,   default=128)
    p.add_argument('--lr',           type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--num_workers',  type=int,   default=4)
    p.add_argument('--seed',         type=int,   default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    train_ds = SpeechCommandsDataset(args.data_dir, split='train', augment=True)
    val_ds   = SpeechCommandsDataset(args.data_dir, split='val',   augment=False)
    test_ds  = SpeechCommandsDataset(args.data_dir, split='test',  augment=False)
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}  Classes: {NUM_CLASSES}")

    loader_kw = dict(
        batch_size=args.batch,
        num_workers=args.num_workers,
        pin_memory=device.type == 'cuda',
        persistent_workers=args.num_workers > 0,
    )
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kw)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kw)

    model     = Res15(num_classes=NUM_CLASSES).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    out_path = Path(args.out_dir) / 'best_res15_speech.pth'

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for spec, labels in train_loader:
            spec, labels = spec.to(device), labels.to(device)
            loss = criterion(model(spec), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * spec.size(0)

        scheduler.step()
        avg_loss = total_loss / len(train_ds)
        val_acc  = evaluate(model, val_loader, device)

        marker = ''
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), out_path)
            marker = f'  -> saved'

        print(f"Epoch {epoch:3d}/{args.epochs}  loss={avg_loss:.4f}  "
              f"val_acc={val_acc:.2f}%  lr={scheduler.get_last_lr()[0]:.2e}{marker}")

    # Final evaluation on test set with best model
    print(f"\nBest val accuracy: {best_val_acc:.2f}%")
    print("Loading best model for test evaluation...")
    model.load_state_dict(torch.load(out_path, map_location=device, weights_only=True))
    test_acc = evaluate(model, test_loader, device)
    print(f"Test accuracy: {test_acc:.2f}%\n")

    print("Per-class test accuracy:")
    per_class = evaluate_per_class(model, test_loader, device)
    for name, acc in sorted(per_class.items(), key=lambda x: x[1]):
        print(f"  {name:<12} {acc:5.1f}%")


if __name__ == '__main__':
    main()
