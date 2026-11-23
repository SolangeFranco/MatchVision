"""Loading and validation for benchmark configuration files."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from importlib.resources import files

import yaml
from jsonschema import Draft202012Validator


class ConfigurationError(ValueError):
    """Raised when an experiment configuration is missing or invalid."""


def load_benchmark_config(path: str | Path) -> dict[str, Any]:
    """Load YAML and validate it against the repository JSON Schema."""
    config_path = Path(path)
    schema_path = files("matchvision").joinpath(
        "schemas", "benchmark.schema.json"
    )
    try:
        with config_path.open("r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
    except (OSError, yaml.YAMLError) as error:
        raise ConfigurationError(f"Could not load config {config_path}: {error}") from error

    import json

    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConfigurationError(f"Could not load schema {schema_path}: {error}") from error

    errors = sorted(
        Draft202012Validator(schema).iter_errors(config),
        key=lambda item: tuple(str(part) for part in item.absolute_path),
    )
    if errors:
        details = "; ".join(
            f"{'.'.join(str(part) for part in error.absolute_path) or '<root>'}: "
            f"{error.message}"
            for error in errors
        )
        raise ConfigurationError(f"Invalid benchmark configuration: {details}")

    return config
