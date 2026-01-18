"""Configuration management for CliffCast.

Provides YAML-based configuration loading with validation and override support.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml
from omegaconf import DictConfig, OmegaConf

from src.utils.logging import get_logger

logger = get_logger(__name__)


# Required configuration fields for validation
REQUIRED_MODEL_FIELDS = [
    "d_model",
    "n_heads",
    "n_layers_transect",
    "n_layers_env",
    "n_layers_fusion",
    "dropout",
    "n_point_features",
    "n_meta_features",
    "n_wave_features",
    "n_precip_features",
]

REQUIRED_DATA_FIELDS = [
    "n_transect_points",
    "wave_history_days",
    "wave_timestep_hours",
    "precip_history_days",
    "precip_timestep_hours",
]

REQUIRED_TRAINING_FIELDS = [
    "batch_size",
    "learning_rate",
    "max_epochs",
]


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""
    pass


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a YAML file.

    Args:
        path: Path to YAML file

    Returns:
        Dictionary containing YAML contents

    Raises:
        FileNotFoundError: If YAML file doesn't exist
        yaml.YAMLError: If YAML is malformed
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML file {path}: {e}")

    return config if config is not None else {}


def load_config(
    config_path: Union[str, Path],
    overrides: Optional[Dict[str, Any]] = None,
    validate: bool = True,
) -> DictConfig:
    """Load configuration from YAML file with optional overrides.

    Args:
        config_path: Path to main config YAML file
        overrides: Optional dictionary of config overrides (e.g., {'model.d_model': 512})
        validate: Whether to validate required fields (default: True)

    Returns:
        OmegaConf DictConfig object with merged configuration

    Raises:
        ConfigValidationError: If validation fails and validate=True
        FileNotFoundError: If config file doesn't exist

    Example:
        >>> cfg = load_config('configs/default.yaml')
        >>> print(cfg.model.d_model)
        256
        >>> cfg = load_config('configs/default.yaml', overrides={'model.d_model': 512})
        >>> print(cfg.model.d_model)
        512
    """
    config_path = Path(config_path)
    logger.info(f"Loading config from: {config_path}")

    # Load base config
    base_config = load_yaml(config_path)
    cfg = OmegaConf.create(base_config)

    # Apply overrides if provided
    if overrides:
        logger.info(f"Applying {len(overrides)} config overrides")
        override_cfg = OmegaConf.create(overrides)
        cfg = OmegaConf.merge(cfg, override_cfg)

    # Validate if requested
    if validate:
        validate_config(cfg)

    logger.info("Configuration loaded successfully")
    return cfg


def validate_config(cfg: DictConfig) -> None:
    """Validate that required configuration fields are present.

    Args:
        cfg: Configuration to validate

    Raises:
        ConfigValidationError: If required fields are missing
    """
    missing_fields = []

    # Check model config
    if "model" in cfg:
        for field in REQUIRED_MODEL_FIELDS:
            if field not in cfg.model:
                missing_fields.append(f"model.{field}")
    else:
        missing_fields.append("model")

    # Check data config
    if "data" in cfg:
        for field in REQUIRED_DATA_FIELDS:
            if field not in cfg.data:
                missing_fields.append(f"data.{field}")
    else:
        missing_fields.append("data")

    # Check training config
    if "training" in cfg:
        for field in REQUIRED_TRAINING_FIELDS:
            if field not in cfg.training:
                missing_fields.append(f"training.{field}")
    else:
        missing_fields.append("training")

    if missing_fields:
        raise ConfigValidationError(
            f"Missing required config fields: {', '.join(missing_fields)}"
        )

    logger.debug("Configuration validation passed")


def merge_configs(
    base_config_path: Union[str, Path],
    override_config_path: Union[str, Path],
) -> DictConfig:
    """Merge two configuration files, with override taking precedence.

    Useful for config inheritance (e.g., phase2 inherits from phase1).

    Args:
        base_config_path: Path to base config file
        override_config_path: Path to override config file

    Returns:
        Merged OmegaConf DictConfig

    Example:
        >>> # phase2_config.yaml inherits from default.yaml but changes some values
        >>> cfg = merge_configs('configs/default.yaml', 'configs/phase2.yaml')
    """
    logger.info(f"Merging configs: {base_config_path} <- {override_config_path}")

    base = load_yaml(base_config_path)
    override = load_yaml(override_config_path)

    base_cfg = OmegaConf.create(base)
    override_cfg = OmegaConf.create(override)

    merged = OmegaConf.merge(base_cfg, override_cfg)

    logger.info("Configs merged successfully")
    return merged


def save_config(cfg: DictConfig, path: Union[str, Path]) -> None:
    """Save configuration to YAML file.

    Useful for saving the exact config used for a training run.

    Args:
        cfg: Configuration to save
        path: Output path for YAML file

    Example:
        >>> cfg = load_config('configs/default.yaml')
        >>> save_config(cfg, 'outputs/run_001/config.yaml')
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        OmegaConf.save(cfg, f)

    logger.info(f"Configuration saved to: {path}")


def config_to_dict(cfg: DictConfig) -> Dict[str, Any]:
    """Convert OmegaConf DictConfig to plain Python dictionary.

    Args:
        cfg: OmegaConf DictConfig

    Returns:
        Plain Python dictionary

    Example:
        >>> cfg = load_config('configs/default.yaml')
        >>> d = config_to_dict(cfg)
        >>> isinstance(d, dict)
        True
    """
    return OmegaConf.to_container(cfg, resolve=True)
