import os
import re
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(:-([^}]*))?\}")


def _expand_env_string(value: str) -> str:
    def repl(match: re.Match) -> str:
        name = match.group(1)
        default = match.group(3)
        if name in os.environ:
            return os.environ[name]
        if default is not None:
            return default
        return match.group(0)

    return _ENV_PATTERN.sub(repl, os.path.expandvars(value))


def expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return _expand_env_string(value)
    if isinstance(value, list):
        return [expand_env(v) for v in value]
    if isinstance(value, dict):
        return {k: expand_env(v) for k, v in value.items()}
    return value


def load_config(path: str, overrides: Mapping[str, Any] = None) -> Dict[str, Any]:
    with Path(path).open() as f:
        cfg = yaml.safe_load(f)
    cfg = expand_env(cfg)
    if overrides:
        for dotted, value in overrides.items():
            node = cfg
            keys = dotted.split(".")
            for key in keys[:-1]:
                node = node.setdefault(key, {})
            node[keys[-1]] = value
    validate_config(cfg)
    return cfg


def validate_config(cfg: Mapping[str, Any]) -> None:
    required = ["experiment", "data", "model", "training", "optimization", "temporal"]
    missing = [key for key in required if key not in cfg]
    if missing:
        raise ValueError(f"Missing config sections: {missing}")
    if cfg["data"].get("dataset") != "cifar100":
        raise ValueError("Feasible CaSSLe PoC currently supports only CIFAR-100")
    if cfg["model"].get("method") not in {"simclr", "byol", "barlow_twins"}:
        raise ValueError("method must be one of simclr, byol, barlow_twins")
    if cfg["model"].get("method") != "simclr" and cfg["training"].get("train_method", True):
        raise ValueError("Only SimCLR is fully supported for training in this PoC")
    for key in ("support_batch_size", "query_batch_size"):
        if int(cfg["data"][key]) < 2:
            raise ValueError(f"{key} must be at least 2")
    if int(cfg["training"]["num_tasks"]) != 5:
        raise ValueError("CIFAR-100 PoC expects 5 class-incremental tasks")


def config_with_metadata(cfg: Mapping[str, Any], commit: str) -> Dict[str, Any]:
    payload = dict(cfg)
    payload["git_commit"] = commit
    return payload

