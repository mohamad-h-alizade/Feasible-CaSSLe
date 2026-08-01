from dataclasses import dataclass
from typing import Dict, Tuple

import torch

from src.cassle_compat import barlow_loss_func, byol_loss_func, simclr_distill_loss_func


def derangement(n: int, device: torch.device = None) -> torch.Tensor:
    if n < 2:
        raise ValueError("A true derangement requires at least two samples")
    device = device or torch.device("cpu")
    for _ in range(64):
        perm = torch.randperm(n, device=device)
        if not torch.any(perm == torch.arange(n, device=device)):
            return perm
    return torch.roll(torch.arange(n, device=device), shifts=1)


@dataclass
class TemporalLossAdapter:
    method: str
    temperature: float = 0.2
    barlow_lamb: float = 5e-3
    barlow_scale_loss: float = 0.025

    @classmethod
    def from_config(cls, cfg: Dict) -> "TemporalLossAdapter":
        method = cfg["model"]["method"]
        temporal = cfg["temporal"]
        if method == "simclr":
            return cls(
                method=method,
                temperature=float(temporal.get("distill_temperature", 0.2)),
            )
        if method == "barlow_twins":
            barlow_cfg = cfg["model"]["barlow_twins"]
            return cls(
                method=method,
                barlow_lamb=float(temporal.get("distill_barlow_lamb", barlow_cfg.get("lamb", 5e-3))),
                barlow_scale_loss=float(
                    temporal.get("distill_scale_loss", barlow_cfg.get("scale_loss", 0.025))
                ),
            )
        if method == "byol":
            return cls(method=method)
        raise ValueError(f"Unsupported temporal loss method: {method}")

    def __call__(
        self,
        p1: torch.Tensor,
        p2: torch.Tensor,
        frozen_z1: torch.Tensor,
        frozen_z2: torch.Tensor,
    ) -> torch.Tensor:
        if self.method == "simclr":
            return (
                simclr_distill_loss_func(p1, p2, frozen_z1, frozen_z2, self.temperature)
                + simclr_distill_loss_func(frozen_z1, frozen_z2, p1, p2, self.temperature)
            ) / 2
        if self.method == "byol":
            return (byol_loss_func(p1, frozen_z1) + byol_loss_func(p2, frozen_z2)) / 2
        if self.method == "barlow_twins":
            return (
                barlow_loss_func(
                    p1,
                    frozen_z1,
                    lamb=self.barlow_lamb,
                    scale_loss=self.barlow_scale_loss,
                )
                + barlow_loss_func(
                    p2,
                    frozen_z2,
                    lamb=self.barlow_lamb,
                    scale_loss=self.barlow_scale_loss,
                )
            ) / 2
        raise ValueError(f"Unsupported temporal loss method: {self.method}")

    def deranged(
        self,
        p1: torch.Tensor,
        p2: torch.Tensor,
        frozen_z1: torch.Tensor,
        frozen_z2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        perm = derangement(frozen_z1.size(0), frozen_z1.device)
        return self(p1, p2, frozen_z1[perm], frozen_z2[perm]), perm


def normalized_reconstruction(raw_loss: torch.Tensor, d_good: float, d_chance: float, eps: float):
    scale = max(float(d_chance) - float(d_good), float(eps))
    return (raw_loss - raw_loss.new_tensor(float(d_good))) / raw_loss.new_tensor(scale)
