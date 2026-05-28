"""Configuration loader. One source of truth for all session parameters."""
from pathlib import Path
import yaml


def load_config(path: str | Path = "config.yaml") -> dict:
    """Load YAML config from disk. Returns a plain dict.

    Args:
        path: Path to config file. Defaults to 'config.yaml' in current dir.

    Returns:
        Nested dict matching the YAML structure.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        yaml.YAMLError: If config is malformed.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path.absolute()}")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    return config


if __name__ == "__main__":
    # Smoke test: load and pretty-print the config
    import json

    cfg = load_config()
    print(json.dumps(cfg, indent=2))
