"""
PRISM Configuration Validator

ZERO DEFAULTS POLICY: All parameters must be explicitly set.

Usage:
    from prism.config.validator import ConfigurationError, validate_required

    # In load_config():
    validate_required(config, ['window_size', 'stride'], 'vector', config_path)
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


class ConfigurationError(Exception):
    """
    Raised when required configuration is missing.

    PRISM enforces a ZERO DEFAULTS policy. All parameters must be
    explicitly set in config. This exception provides clear error
    messages telling users exactly what to configure.
    """
    pass


# Required fields per pipeline stage
REQUIRED_FIELDS = {
    'vector': [
        'window_size',
        'stride',
    ],
    'geometry': [
        'window_size',
        'stride',
        'min_samples_geometry',
    ],
    'dynamics': [
        'window_size',
        'stride',
        'min_samples_dynamics',
    ],
    'physics': [
        'window_size',
        'stride',
    ],
}


def validate_required(
    config: Dict[str, Any],
    required_keys: List[str],
    stage: str,
    config_path: Optional[Path] = None,
) -> None:
    """
    Validate that all required configuration keys are present.

    Args:
        config: Configuration dictionary
        required_keys: List of keys that must be present and not None
        stage: Pipeline stage name (for error message)
        config_path: Path to config file (for error message)

    Raises:
        ConfigurationError: If any required key is missing or None
    """
    missing = []

    for key in required_keys:
        if key not in config or config[key] is None:
            missing.append(key)

    if missing:
        location = f"File: {config_path}\n" if config_path else ""
        raise ConfigurationError(
            f"\n{'='*60}\n"
            f"CONFIGURATION ERROR: Missing required parameters\n"
            f"{'='*60}\n"
            f"{location}"
            f"Pipeline stage: {stage}\n\n"
            f"Missing fields:\n"
            f"{''.join(f'  - {k}' + chr(10) for k in missing)}\n"
            f"PRISM enforces a ZERO DEFAULTS policy.\n"
            f"All parameters must be explicitly set.\n\n"
            f"Add to your config.yaml:\n"
            f"{''.join(f'  {k}: <value>' + chr(10) for k in missing)}\n"
            f"{'='*60}"
        )


def validate_stage(config: Dict[str, Any], stage: str, config_path: Optional[Path] = None) -> None:
    """
    Validate configuration for a specific pipeline stage.

    Uses predefined required fields per stage.

    Args:
        config: Configuration dictionary
        stage: One of 'vector', 'geometry', 'dynamics', 'physics'
        config_path: Path to config file (for error message)

    Raises:
        ConfigurationError: If stage unknown or required fields missing
    """
    if stage not in REQUIRED_FIELDS:
        raise ConfigurationError(f"Unknown pipeline stage: {stage}")

    validate_required(config, REQUIRED_FIELDS[stage], stage, config_path)


def validate_or_die(config: Dict[str, Any], stage: str, config_path: Optional[Path] = None) -> None:
    """
    Validate configuration. Exit with error code 1 if invalid.

    Use this at entry points for clear error messages and clean exit.

    Args:
        config: Configuration dictionary
        stage: Pipeline stage name
        config_path: Path to config file (for error message)
    """
    try:
        validate_stage(config, stage, config_path)
    except ConfigurationError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)


def require_key(config: Dict[str, Any], key: str, stage: str = "") -> Any:
    """
    Get a required configuration value.

    Unlike dict.get(), this NEVER returns a default value.
    Raises ConfigurationError if key is missing or None.

    Args:
        config: Configuration dictionary
        key: Key to retrieve
        stage: Pipeline stage (for error message)

    Returns:
        The configuration value

    Raises:
        ConfigurationError: If key is missing or None
    """
    if key not in config or config[key] is None:
        raise ConfigurationError(
            f"\n{'='*60}\n"
            f"CONFIGURATION ERROR: {key} not set\n"
            f"{'='*60}\n"
            f"Stage: {stage}\n\n"
            f"{key} is REQUIRED.\n"
            f"Set it in your config.yaml:\n\n"
            f"  {key}: <value>\n\n"
            f"NO DEFAULTS. NO FALLBACKS. Configure your domain.\n"
            f"{'='*60}"
        )

    return config[key]
