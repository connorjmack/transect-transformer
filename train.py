#!/usr/bin/env python3
"""
Simple training script for CliffCast model.

Usage:
    # Quick test (small model, few epochs)
    python train.py --data data/processed/training_data.npz --epochs 5 --batch-size 16 --debug

    # Full training
    python train.py --data data/processed/training_data.npz --epochs 100 --batch-size 32

Author: CliffCast Project
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import CliffCastDataset, collate_fn, create_dataloaders
from src.models.cliffcast import CliffCast

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# Loss Function
# =============================================================================

class SimpleLoss(nn.Module):
    """
    Simplified loss for Phase 1 training (risk index focus).

    Supports:
    - Risk index: Smooth L1 loss
    - Event class: Cross-entropy with class weights
    - Confidence weighting: Weight samples by confidence score
    """

    def __init__(
        self,
        weight_risk: float = 1.0,
        weight_event_class: float = 0.5,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.weight_risk = weight_risk
        self.weight_event_class = weight_event_class
        self.class_weights = class_weights

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute losses."""
        losses = {}
        total_loss = torch.tensor(0.0, device=predictions['risk_index'].device)

        # Risk index loss (Smooth L1)
        if 'risk_index' in predictions:
            # Weight by confidence
            confidence = targets.get('confidence', torch.ones_like(targets['risk_index']))
            risk_diff = F.smooth_l1_loss(predictions['risk_index'], targets['risk_index'], reduction='none')
            loss_risk = (risk_diff * confidence).mean()
            losses['loss_risk'] = loss_risk
            total_loss = total_loss + self.weight_risk * loss_risk

        # Event class loss (Cross-entropy)
        if 'failure_mode_logits' in predictions and 'event_class' in targets:
            weight = self.class_weights.to(predictions['failure_mode_logits'].device) if self.class_weights is not None else None
            # Only 4 classes now (stable, minor, major, failure)
            logits = predictions['failure_mode_logits'][:, :4]  # Take first 4 classes
            loss_class = F.cross_entropy(logits, targets['event_class'], weight=weight)
            losses['loss_event_class'] = loss_class
            total_loss = total_loss + self.weight_event_class * loss_class

        losses['loss'] = total_loss
        return losses


# =============================================================================
# Training Functions
# =============================================================================

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_clip: float = 1.0,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_risk_loss = 0.0
    n_batches = 0

    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        # Move to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # Forward pass
        outputs = model(
            point_features=batch['point_features'],
            metadata=batch['metadata'],
            distances=batch['distances'],
            wave_features=batch['wave_features'],
            atmos_features=batch['atmos_features'],
            wave_doy=batch['wave_doy'],
            atmos_doy=batch['atmos_doy'],
        )

        # Compute loss
        losses = criterion(outputs, batch)
        loss = losses['loss']

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        total_risk_loss += losses.get('loss_risk', torch.tensor(0.0)).item()
        n_batches += 1

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return {
        'loss': total_loss / n_batches,
        'loss_risk': total_risk_loss / n_batches,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Validate model."""
    model.eval()
    total_loss = 0.0
    total_risk_loss = 0.0
    all_preds = []
    all_targets = []
    n_batches = 0

    for batch in tqdm(dataloader, desc="Validating"):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        outputs = model(
            point_features=batch['point_features'],
            metadata=batch['metadata'],
            distances=batch['distances'],
            wave_features=batch['wave_features'],
            atmos_features=batch['atmos_features'],
            wave_doy=batch['wave_doy'],
            atmos_doy=batch['atmos_doy'],
        )

        losses = criterion(outputs, batch)

        total_loss += losses['loss'].item()
        total_risk_loss += losses.get('loss_risk', torch.tensor(0.0)).item()

        all_preds.append(outputs['risk_index'].cpu())
        all_targets.append(batch['risk_index'].cpu())
        n_batches += 1

    # Compute metrics
    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)

    # MAE for risk index
    mae = (preds - targets).abs().mean().item()

    # Correlation
    if preds.std() > 0 and targets.std() > 0:
        corr = torch.corrcoef(torch.stack([preds, targets]))[0, 1].item()
    else:
        corr = 0.0

    return {
        'loss': total_loss / n_batches,
        'loss_risk': total_risk_loss / n_batches,
        'mae': mae,
        'correlation': corr,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train CliffCast model")
    parser.add_argument('--data', type=str, default='data/processed/training_data.npz', help='Path to training data')
    parser.add_argument('--output', type=str, default='checkpoints', help='Output directory for checkpoints')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--grad-clip', type=float, default=1.0, help='Gradient clipping')
    parser.add_argument('--debug', action='store_true', help='Use small model for debugging')
    parser.add_argument('--num-workers', type=int, default=0, help='Data loader workers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    logger.info(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info(f"Loading data from {args.data}")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_path=args.data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    logger.info(f"Train: {len(train_loader.dataset)} samples, Val: {len(val_loader.dataset)} samples, Test: {len(test_loader.dataset)} samples")

    # Get class weights for imbalanced data
    class_weights = train_loader.dataset.get_class_weights()
    logger.info(f"Class weights: {class_weights.tolist()}")

    # Create model
    if args.debug:
        # Small model for debugging
        model = CliffCast(
            d_model=64,
            n_heads=4,
            n_layers_spatial=1,
            n_layers_temporal=1,
            n_layers_env=1,
            n_layers_fusion=1,
            dropout=0.1,
            enable_risk=True,
            enable_retreat=False,
            enable_collapse=False,
            enable_failure_mode=True,
            n_failure_modes=4,  # 4 event classes
        )
        logger.info("Using DEBUG model (small)")
    else:
        # Full model
        model = CliffCast(
            d_model=256,
            n_heads=8,
            n_layers_spatial=2,
            n_layers_temporal=2,
            n_layers_env=3,
            n_layers_fusion=2,
            dropout=0.1,
            enable_risk=True,
            enable_retreat=False,
            enable_collapse=False,
            enable_failure_mode=True,
            n_failure_modes=4,  # 4 event classes
        )
        logger.info("Using FULL model")

    model = model.to(device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params:,}")

    # Loss function
    criterion = SimpleLoss(
        weight_risk=1.0,
        weight_event_class=0.5,
        class_weights=class_weights,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr / 100,
    )

    # Training loop
    best_val_loss = float('inf')
    best_epoch = 0

    logger.info(f"\nStarting training for {args.epochs} epochs...")

    for epoch in range(1, args.epochs + 1):
        logger.info(f"\nEpoch {epoch}/{args.epochs}")

        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, args.grad_clip)
        logger.info(f"  Train - Loss: {train_metrics['loss']:.4f}, Risk Loss: {train_metrics['loss_risk']:.4f}")

        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        logger.info(f"  Val   - Loss: {val_metrics['loss']:.4f}, Risk Loss: {val_metrics['loss_risk']:.4f}, MAE: {val_metrics['mae']:.4f}, Corr: {val_metrics['correlation']:.4f}")

        # Step scheduler
        scheduler.step()

        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_epoch = epoch

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'val_mae': val_metrics['mae'],
                'args': vars(args),
            }
            torch.save(checkpoint, output_dir / 'best_model.pt')
            logger.info(f"  Saved new best model (val_loss={best_val_loss:.4f})")

        # Save periodic checkpoint
        if epoch % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'args': vars(args),
            }
            torch.save(checkpoint, output_dir / f'checkpoint_epoch_{epoch}.pt')

    # Final evaluation on test set
    logger.info(f"\nTraining complete. Best model from epoch {best_epoch}")

    # Load best model
    checkpoint = torch.load(output_dir / 'best_model.pt', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Evaluate on test set
    test_metrics = validate(model, test_loader, criterion, device)
    logger.info(f"\nTest Results:")
    logger.info(f"  Loss: {test_metrics['loss']:.4f}")
    logger.info(f"  Risk MAE: {test_metrics['mae']:.4f}")
    logger.info(f"  Risk Correlation: {test_metrics['correlation']:.4f}")

    # Save final checkpoint with test metrics
    checkpoint['test_metrics'] = test_metrics
    torch.save(checkpoint, output_dir / 'best_model.pt')

    logger.info(f"\nCheckpoints saved to {output_dir}/")


if __name__ == '__main__':
    main()
