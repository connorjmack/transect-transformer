"""Tests for utility functions."""

import pytest
from pathlib import Path

from src.utils.config import (
    ConfigValidationError,
    load_config,
    merge_configs,
    save_config,
    validate_config,
    config_to_dict,
)


def test_load_default_config():
    """Test loading default configuration."""
    cfg = load_config("configs/default.yaml")

    # Check model config
    assert cfg.model.d_model == 256
    assert cfg.model.n_heads == 8
    assert cfg.model.n_layers_transect == 4

    # Check data config
    assert cfg.data.n_transect_points == 128
    assert cfg.data.wave_history_days == 90

    # Check training config
    assert cfg.training.batch_size == 32
    assert cfg.training.learning_rate == 1.0e-4


def test_load_phase1_config():
    """Test loading Phase 1 configuration (risk only)."""
    cfg = load_config("configs/phase1_risk_only.yaml")

    # Check heads are configured correctly
    assert cfg.model.enable_risk is True
    assert cfg.model.enable_retreat is False
    assert cfg.model.enable_collapse is False
    assert cfg.model.enable_failure_mode is False

    # Check paths are phase-specific
    assert "phase1" in cfg.paths.checkpoint_dir


def test_config_validation():
    """Test configuration validation."""
    # Valid config should pass
    cfg = load_config("configs/default.yaml", validate=True)
    validate_config(cfg)  # Should not raise

    # Invalid config should fail
    from omegaconf import OmegaConf

    invalid_cfg = OmegaConf.create({"model": {"d_model": 256}})  # Missing required fields
    with pytest.raises(ConfigValidationError):
        validate_config(invalid_cfg)


def test_config_overrides():
    """Test configuration overrides."""
    overrides = {"model.d_model": 512, "training.batch_size": 64}
    cfg = load_config("configs/default.yaml", overrides=overrides)

    assert cfg.model.d_model == 512
    assert cfg.training.batch_size == 64
    # Other values should remain unchanged
    assert cfg.model.n_heads == 8


def test_config_to_dict():
    """Test converting OmegaConf to dict."""
    cfg = load_config("configs/default.yaml")
    d = config_to_dict(cfg)

    assert isinstance(d, dict)
    assert d["model"]["d_model"] == 256
    assert d["training"]["batch_size"] == 32


def test_save_and_load_config(tmp_path):
    """Test saving and loading configuration."""
    cfg = load_config("configs/default.yaml")

    save_path = tmp_path / "test_config.yaml"
    save_config(cfg, save_path)

    assert save_path.exists()

    # Load the saved config
    loaded_cfg = load_config(save_path)
    assert loaded_cfg.model.d_model == cfg.model.d_model
    assert loaded_cfg.training.batch_size == cfg.training.batch_size


def test_merge_configs():
    """Test merging two configurations."""
    # phase1 should have different head settings than default
    cfg = load_config("configs/phase1_risk_only.yaml")

    assert cfg.model.enable_risk is True
    assert cfg.model.enable_retreat is False

    # Check that other settings are still present
    assert cfg.model.d_model == 256
    assert cfg.training.batch_size == 32


def test_missing_config_file():
    """Test loading non-existent config file."""
    with pytest.raises(FileNotFoundError):
        load_config("configs/nonexistent.yaml")


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])
