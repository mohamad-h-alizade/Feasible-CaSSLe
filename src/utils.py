import csv
import json
import os
import random
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, Iterator, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def make_run_dir(output_root: str, name: str) -> Path:
    root = Path(output_root).expanduser()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = root / name / stamp
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing result directory: {path}")
    path.mkdir(parents=True, exist_ok=False)
    return path


def save_json(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)


def append_csv(path: Path, row: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)
        return

    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    expanded = False
    for key in row.keys():
        if key not in fieldnames:
            fieldnames.append(key)
            expanded = True

    if expanded:
        rows.append(dict(row))
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    else:
        with path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(row)


def assert_finite_tensor(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"{name} contains NaN or infinity")


def assert_finite_number(name: str, value: float) -> None:
    if not np.isfinite(value):
        raise FloatingPointError(f"{name} is not finite: {value}")


def module_trainable_parameters(module: nn.Module) -> Iterator[nn.Parameter]:
    for param in module.parameters():
        if param.requires_grad:
            yield param


def set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for param in module.parameters():
        param.requires_grad_(enabled)


@contextmanager
def frozen_params(module: nn.Module):
    states = [param.requires_grad for param in module.parameters()]
    set_requires_grad(module, False)
    try:
        yield
    finally:
        for param, state in zip(module.parameters(), states):
            param.requires_grad_(state)


@contextmanager
def eval_mode(module: nn.Module):
    was_training = module.training
    module.eval()
    try:
        yield
    finally:
        module.train(was_training)


def _bn_buffers(module: nn.Module) -> Dict[str, torch.Tensor]:
    buffers = {}
    for name, submodule in module.named_modules():
        if isinstance(submodule, nn.modules.batchnorm._BatchNorm):
            prefix = f"{name}." if name else ""
            for key in ("running_mean", "running_var", "num_batches_tracked"):
                value = getattr(submodule, key, None)
                if value is not None:
                    buffers[prefix + key] = value.detach().clone()
    return buffers


@contextmanager
def preserve_batchnorm_stats(module: nn.Module):
    """Restore BatchNorm running buffers after a forward pass.

    Support adaptation should not mutate the current encoder's running
    statistics. This context keeps normal train/eval behavior for computation
    but restores all BatchNorm buffers afterward.
    """

    before = _bn_buffers(module)
    try:
        yield
    finally:
        modules = dict(module.named_modules())
        for full_name, value in before.items():
            if "." in full_name:
                module_name, buffer_name = full_name.rsplit(".", 1)
                target = modules[module_name]
            else:
                target = module
                buffer_name = full_name
            getattr(target, buffer_name).copy_(value)


def tensor_fingerprint(tensors: Iterable[torch.Tensor]) -> Sequence[float]:
    return [float(t.detach().double().sum().cpu()) for t in tensors]


def parameters_changed(before: Sequence[torch.Tensor], after: Iterable[torch.Tensor]) -> bool:
    for old, new in zip(before, after):
        if not torch.equal(old, new.detach().cpu()):
            return True
    return False


def clone_parameters(module: nn.Module) -> Sequence[torch.Tensor]:
    return [p.detach().cpu().clone() for p in module.parameters()]


def peak_memory_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return float(torch.cuda.max_memory_allocated(device) / (1024**2))


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def progress_enabled(cfg: Mapping) -> bool:
    return bool(cfg.get("logging", {}).get("progress", True))


def progress_interval(cfg: Mapping) -> int:
    return max(1, int(cfg.get("logging", {}).get("progress_interval", 10)))


def progress_print(cfg: Mapping, message: str) -> None:
    if progress_enabled(cfg):
        print(message, flush=True)
