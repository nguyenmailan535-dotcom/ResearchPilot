"""Configuration loading utilities."""

import json
from pathlib import Path

import pydantic
from loguru import logger

from nanobot.config.schema import Config

# Global variable to store current config path (for multi-instance support)
_current_config_path: Path | None = None


def set_config_path(path: Path) -> None:
    """Set the current config path (used to derive data directory)."""
    global _current_config_path
    _current_config_path = path


def get_config_path() -> Path:
    """Get the configuration file path."""
    if _current_config_path:
        return _current_config_path
    return Path.home() / ".nanobot" / "config.json"


def load_config(config_path: Path | None = None) -> Config:
    """
    Load configuration from file or create default.

    Args:
        config_path: Optional path to config file. Uses default if not provided.

    Returns:
        Loaded configuration object.
    """
    path = config_path or get_config_path()

    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            data = _migrate_config(data)
            return Config.model_validate(data)
        except (json.JSONDecodeError, ValueError, pydantic.ValidationError) as e:
            logger.warning(f"Failed to load config from {path}: {e}")
            logger.warning("Using default configuration.")

    return Config()


def save_config(config: Config, config_path: Path | None = None) -> None:
    """
    Save configuration to file.

    Args:
        config: Configuration to save.
        config_path: Optional path to save to. Uses default if not provided.
    """
    path = config_path or get_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    data = config.model_dump(mode="json", by_alias=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _migrate_config(data: dict) -> dict:
    """Migrate old config formats to current."""
    # Newer nanobot configs serialize unset provider keys as null, while this runtime uses an
    # empty string. Normalize them before Pydantic validation so one unused provider cannot make
    # the entire configuration fall back to defaults.
    providers = data.get("providers", {})
    if isinstance(providers, dict):
        for provider in providers.values():
            if not isinstance(provider, dict):
                continue
            for key in ("apiKey", "api_key"):
                if key in provider and provider[key] is None:
                    provider[key] = ""

    # Newer releases can select a named model preset. Materialize the selected preset into the
    # fields understood by the v0.1.4 AgentDefaults schema.
    agents = data.get("agents", {})
    defaults = agents.get("defaults", {}) if isinstance(agents, dict) else {}
    presets = data.get("modelPresets", data.get("model_presets", {}))
    if isinstance(defaults, dict) and isinstance(presets, dict):
        preset_name = defaults.get("modelPreset", defaults.get("model_preset"))
        preset = presets.get(preset_name) if isinstance(preset_name, str) else None
        if isinstance(preset, dict):
            for key in (
                "model",
                "provider",
                "maxTokens",
                "contextWindowTokens",
                "temperature",
                "reasoningEffort",
            ):
                if key in preset:
                    defaults[key] = preset[key]

    # These newer top-level sections have no v0.1.4 runtime equivalent. Their relevant model
    # values were materialized above; ignoring the remaining metadata is safe.
    data.pop("modelPresets", None)
    data.pop("model_presets", None)
    data.pop("transcription", None)

    # Move tools.exec.restrictToWorkspace → tools.restrictToWorkspace
    tools = data.get("tools", {})
    exec_cfg = tools.get("exec", {})
    if "restrictToWorkspace" in exec_cfg and "restrictToWorkspace" not in tools:
        tools["restrictToWorkspace"] = exec_cfg.pop("restrictToWorkspace")
    return data
