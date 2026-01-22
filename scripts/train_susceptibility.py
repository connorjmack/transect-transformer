#!/usr/bin/env python3
"""
Train CliffCast susceptibility classifier.

This script trains the 5-class erosion mode classifier using:
- Weighted cross-entropy loss with label smoothing
- AdamW optimizer with cosine learning rate schedule
- Gradient clipping for stability
- Early stopping based on validation loss

Usage:
    # Train with config file
    python scripts/train_susceptibility.py --config configs/susceptibility_v1.yaml

    # Train with custom data path
    python scripts/train_susceptibility.py --config configs/susceptibility_v1.yaml --train-data data/processed/training_data.npz

    # Quick test with small model
    python scripts/train_susceptibility.py --config configs/susceptibility_v1.yaml --debug
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.cliffcast import CliffCast
from src.data.susceptibility_dataset import SusceptibilityDataset, create_data_loaders
from src.training.susceptibility_loss import SusceptibilityLoss
from src.utils.logging import setup_logger

logger = setup_logger(__name__, level="INFO")


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def create_model(config: dict, device: torch.device) -> CliffCast:
    """Create CliffCast model from config."""
    model_cfg = config.get('model', {})

    model = CliffCast(
        d_model=model_cfg.get('d_model', 256),
        n_heads=model_cfg.get('n_heads', 8),
        dropout=model_cfg.get('dropout', 0.1),
        n_layers_spatial=model_cfg.get('n_layers_spatial', 2),
        n_layers_temporal=model_cfg.get('n_layers_temporal', 2),
        max_timesteps=model_cfg.get('max_timesteps', 20),
        n_point_features=model_cfg.get('n_point_features', 7),
        n_metadata_features=model_cfg.get('n_metadata_features', 12),
        n_layers_env=model_cfg.get('n_layers_env', 3),
        n_wave_features=model_cfg.get('n_wave_features', 4),
        n_atmos_features=model_cfg.get('n_atmos_features', 24),
        n_layers_fusion=model_cfg.get('n_layers_fusion', 2),
        # Only enable failure mode head for susceptibility classification
        enable_risk=False,
        enable_retreat=False,
        enable_collapse=False,
        enable_failure_mode=True,
        n_failure_modes=5,
    )

    return model.to(device)


def compute_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> Dict[str, float]:
    """Compute classification metrics."""
    preds = logits.argmax(dim=-1)
    correct = (preds == targets).float()
    accuracy = correct.mean().item()

    # Per-class accuracy
    metrics = {'accuracy': accuracy}
    for c in range(5):
        mask = targets == c
        if mask.sum() > 0:
            class_acc = correct[mask].mean().item()
            metrics[f'acc_class_{c}'] = class_acc

    return metrics


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    gradient_clip: float = 1.0,
    use_amp: bool = False,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    class_correct = np.zeros(5)
    class_total = np.zeros(5)

    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    for batch in train_loader:
        # Move to device
        point_features = batch['point_features'].to(device)
        metadata = batch['metadata'].to(device)
        distances = batch['distances'].to(device)
        wave_features = batch['wave_features'].to(device)
        wave_doy = batch['wave_doy'].to(device)
        atmos_features = batch['atmos_features'].to(device)
        atmos_doy = batch['atmos_doy'].to(device)
        targets = batch['event_class'].to(device)

        optimizer.zero_grad()

        # Forward pass
        if use_amp:
            with torch.cuda.amp.autocast():
                outputs = model(
                    point_features=point_features,
                    metadata=metadata,
                    distances=distances,
                    wave_features=wave_features,
                    wave_doy=wave_doy,
                    atmos_features=atmos_features,
                    atmos_doy=atmos_doy,
                )
                logits = outputs['failure_mode_logits']
                loss = criterion(logits, targets)
        else:
            outputs = model(
                point_features=point_features,
                metadata=metadata,
                distances=distances,
                wave_features=wave_features,
                wave_doy=wave_doy,
                atmos_features=atmos_features,
                atmos_doy=atmos_doy,
            )
            logits = outputs['failure_mode_logits']
            loss = criterion(logits, targets)

        # Backward pass
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()

        # Track metrics
        total_loss += loss.item() * targets.size(0)
        preds = logits.argmax(dim=-1)
        total_correct += (preds == targets).sum().item()
        total_samples += targets.size(0)

        # Per-class tracking
        for c in range(5):
            mask = targets == c
            class_total[c] += mask.sum().item()
            class_correct[c] += ((preds == targets) & mask).sum().item()

    # Compute epoch metrics
    metrics = {
        'loss': total_loss / total_samples,
        'accuracy': total_correct / total_samples,
    }
    for c in range(5):
        if class_total[c] > 0:
            metrics[f'acc_class_{c}'] = class_correct[c] / class_total[c]

    return metrics


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Validate model."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    class_correct = np.zeros(5)
    class_total = np.zeros(5)

    for batch in val_loader:
        # Move to device
        point_features = batch['point_features'].to(device)
        metadata = batch['metadata'].to(device)
        distances = batch['distances'].to(device)
        wave_features = batch['wave_features'].to(device)
        wave_doy = batch['wave_doy'].to(device)
        atmos_features = batch['atmos_features'].to(device)
        atmos_doy = batch['atmos_doy'].to(device)
        targets = batch['event_class'].to(device)

        # Forward pass
        outputs = model(
            point_features=point_features,
            metadata=metadata,
            distances=distances,
            wave_features=wave_features,
            wave_doy=wave_doy,
            atmos_features=atmos_features,
            atmos_doy=atmos_doy,
        )
        logits = outputs['failure_mode_logits']
        loss = criterion(logits, targets)

        # Track metrics
        total_loss += loss.item() * targets.size(0)
        preds = logits.argmax(dim=-1)
        total_correct += (preds == targets).sum().item()
        total_samples += targets.size(0)

        # Per-class tracking
        for c in range(5):
            mask = targets == c
            class_total[c] += mask.sum().item()
            class_correct[c] += ((preds == targets) & mask).sum().item()

    # Compute epoch metrics
    metrics = {
        'loss': total_loss / total_samples,
        'accuracy': total_correct / total_samples,
    }
    for c in range(5):
        if class_total[c] > 0:
            metrics[f'acc_class_{c}'] = class_correct[c] / class_total[c]

    return metrics


def train(
    config: dict,
    train_path: Path,
    val_path: Optional[Path] = None,
    checkpoint_dir: Path = Path('checkpoints'),
    debug: bool = False,
    subset_fraction: Optional[float] = None,
    max_epochs_override: Optional[int] = None,
):
    """Main training function."""
    # Setup - prefer MPS (Apple Silicon) over CPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    logger.info(f"Using device: {device}")

    train_cfg = config.get('training', {})

    # Debug mode: use small model and subset
    if debug:
        config['model'] = {
            'd_model': 64,
            'n_heads': 4,
            'n_layers_spatial': 1,
            'n_layers_temporal': 1,
            'n_layers_env': 1,
            'n_layers_fusion': 1,
            'n_point_features': 7,
            'n_metadata_features': 12,
            'dropout': 0.1,
        }
        train_cfg['batch_size'] = 8
        train_cfg['max_epochs'] = 3
        subset_fraction = 0.01

    # Override max epochs if specified
    if max_epochs_override is not None:
        train_cfg['max_epochs'] = max_epochs_override

    # Create data loaders
    logger.info(f"Loading training data from {train_path}")
    train_loader, val_loader = create_data_loaders(
        train_path=train_path,
        val_path=val_path,
        batch_size=train_cfg.get('batch_size', 32),
        num_workers=train_cfg.get('num_workers', 4),
        use_weighted_sampler=train_cfg.get('use_weighted_sampler', True),
        subset_fraction=subset_fraction,
    )
    logger.info(f"Training samples: {len(train_loader.dataset)}")
    if val_loader:
        logger.info(f"Validation samples: {len(val_loader.dataset)}")

    # Create model
    logger.info("Creating model")
    model = create_model(config, device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params:,}")

    # Create loss function
    criterion = SusceptibilityLoss(
        label_smoothing=train_cfg.get('label_smoothing', 0.1),
    ).to(device)

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.get('learning_rate', 1e-4),
        weight_decay=train_cfg.get('weight_decay', 1e-5),
    )

    # Create scheduler
    max_epochs = train_cfg.get('max_epochs', 100)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max_epochs,
        eta_min=train_cfg.get('min_lr', 1e-6),
    )

    # Training state
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    patience = train_cfg.get('early_stopping_patience', 15)

    # Create checkpoint directory
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    logger.info("Starting training")
    for epoch in range(max_epochs):
        epoch_start = datetime.now()

        # Train
        train_metrics = train_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            gradient_clip=train_cfg.get('gradient_clip', 1.0),
            use_amp=train_cfg.get('use_amp', False),
        )

        # Validate
        if val_loader:
            val_metrics = validate(model, val_loader, criterion, device)
            val_loss = val_metrics['loss']
        else:
            # Use training loss if no validation set
            val_metrics = train_metrics
            val_loss = train_metrics['loss']

        # Update scheduler
        scheduler.step()

        # Logging
        epoch_time = (datetime.now() - epoch_start).total_seconds()
        lr = optimizer.param_groups[0]['lr']
        logger.info(
            f"Epoch {epoch+1}/{max_epochs} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Train Acc: {train_metrics['accuracy']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"LR: {lr:.2e} | "
            f"Time: {epoch_time:.1f}s"
        )

        # Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0

            # Save best model
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'config': config,
            }
            torch.save(checkpoint, checkpoint_dir / 'best_model.pt')
            logger.info(f"  Saved best model (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

    # Save final model
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'config': config,
    }
    torch.save(checkpoint, checkpoint_dir / 'final_model.pt')

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Training Complete")
    logger.info("=" * 60)
    logger.info(f"Best epoch: {best_epoch + 1}")
    logger.info(f"Best val loss: {best_val_loss:.4f}")
    logger.info(f"Checkpoints saved to: {checkpoint_dir}")

    return model, best_val_loss


def main():
    parser = argparse.ArgumentParser(
        description="Train CliffCast susceptibility classifier"
    )
    parser.add_argument(
        '--config', '-c',
        type=Path,
        default=Path('configs/susceptibility_v1.yaml'),
        help='Path to config YAML file'
    )
    parser.add_argument(
        '--train-data',
        type=Path,
        help='Path to training data NPZ (overrides config)'
    )
    parser.add_argument(
        '--val-data',
        type=Path,
        help='Path to validation data NPZ'
    )
    parser.add_argument(
        '--checkpoint-dir',
        type=Path,
        default=Path('checkpoints'),
        help='Directory to save checkpoints'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Debug mode: small model, few epochs, data subset'
    )
    parser.add_argument(
        '--subset',
        type=float,
        default=None,
        help='Fraction of data to use (0.0-1.0), e.g. 0.1 for 10%%'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Override max epochs from config'
    )

    args = parser.parse_args()

    # Load config
    if args.config.exists():
        config = load_config(args.config)
        logger.info(f"Loaded config from {args.config}")
    else:
        logger.warning(f"Config not found at {args.config}, using defaults")
        config = {}

    # Get data paths
    train_path = args.train_data
    if train_path is None:
        train_path = Path(config.get('data', {}).get('train_path', 'data/processed/training_data_300.npz'))

    val_path = args.val_data
    if val_path is None:
        val_path = config.get('data', {}).get('val_path')
        if val_path:
            val_path = Path(val_path)

    if not train_path.exists():
        logger.error(f"Training data not found: {train_path}")
        return 1

    # Print header
    print("\n" + "=" * 60)
    print("CLIFFCAST SUSCEPTIBILITY TRAINING")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Config: {args.config}")
    print(f"Train data: {train_path}")
    print(f"Val data: {val_path}")
    print(f"Debug mode: {args.debug}")
    print()

    # Train
    train(
        config=config,
        train_path=train_path,
        val_path=val_path,
        checkpoint_dir=args.checkpoint_dir,
        debug=args.debug,
        subset_fraction=args.subset,
        max_epochs_override=args.epochs,
    )

    return 0


if __name__ == '__main__':
    sys.exit(main())
