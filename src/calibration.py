from dataclasses import asdict, dataclass
from typing import Dict, Iterable, Iterator

import torch
from torch import nn

from src.data import SupportQueryBatch, move_batch
from src.models import forward_project
from src.temporal_losses import TemporalLossAdapter
from src.utils import eval_mode, preserve_batchnorm_stats


@dataclass
class CalibrationResult:
    d_good: float
    d_chance: float
    scale: float
    raw_budget: float
    rho: float

    def asdict(self) -> Dict[str, float]:
        return asdict(self)


def support_predictor_step(
    model: nn.Module,
    historical: nn.Module,
    predictor: nn.Module,
    optimizer: torch.optim.Optimizer,
    adapter: TemporalLossAdapter,
    views,
) -> torch.Tensor:
    predictor.train()
    optimizer.zero_grad(set_to_none=True)
    x1, x2 = views
    with torch.no_grad(), preserve_batchnorm_stats(model):
        z1 = forward_project(model, x1).detach()
        z2 = forward_project(model, x2).detach()
    with torch.no_grad(), eval_mode(historical):
        frozen_z1 = forward_project(historical, x1).detach()
        frozen_z2 = forward_project(historical, x2).detach()
    loss = adapter(predictor(z1), predictor(z2), frozen_z1, frozen_z2)
    loss.backward()
    optimizer.step()
    return loss.detach()


@torch.no_grad()
def evaluate_teacher_temporal_batch(
    historical: nn.Module,
    adapter: TemporalLossAdapter,
    views,
) -> Dict[str, float]:
    x1, x2 = views
    with eval_mode(historical):
        frozen_z1 = forward_project(historical, x1)
        frozen_z2 = forward_project(historical, x2)
        good = adapter(frozen_z1, frozen_z2, frozen_z1, frozen_z2)
        chance, perm = adapter.deranged(frozen_z1, frozen_z2, frozen_z1, frozen_z2)
        if torch.any(perm == torch.arange(perm.numel(), device=perm.device)):
            raise AssertionError("Calibration derangement contains fixed points")
    return {"good": float(good.cpu()), "chance": float(chance.cpu())}


def calibrate_task(
    cfg: Dict,
    model: nn.Module,
    historical: nn.Module,
    predictor: nn.Module,
    predictor_optimizer: torch.optim.Optimizer,
    adapter: TemporalLossAdapter,
    batches: Iterator[SupportQueryBatch],
    device: torch.device,
) -> CalibrationResult:
    temporal = cfg["temporal"]
    good_losses = []
    chance_losses = []
    for _ in range(int(temporal.get("calibration_batches", 16))):
        batch = move_batch(next(batches), device)
        values = evaluate_teacher_temporal_batch(
            historical,
            adapter,
            batch.query_views,
        )
        good_losses.append(values["good"])
        chance_losses.append(values["chance"])

    good_tensor = torch.tensor(good_losses)
    chance_tensor = torch.tensor(chance_losses)
    d_good = float(torch.quantile(good_tensor, float(temporal.get("good_quantile", 0.95))).item())
    d_chance = float(torch.median(chance_tensor).item())
    eps = float(temporal.get("epsilon_num", 1e-8))
    if d_chance <= d_good:
        raise RuntimeError(
            "Degenerate temporal calibration: D_chance <= D_good. "
            "Teacher-only chance loss should exceed teacher-self loss. "
            "Try larger query batches/calibration_batches, check the temporal objective, and inspect for collapse."
        )
    scale = max(d_chance - d_good, eps)
    rho = float(temporal.get("rho", 0.05))
    raw_budget = d_good + rho * (d_chance - d_good)
    return CalibrationResult(
        d_good=d_good,
        d_chance=d_chance,
        scale=scale,
        raw_budget=raw_budget,
        rho=rho,
    )
