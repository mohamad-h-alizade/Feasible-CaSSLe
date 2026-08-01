from dataclasses import asdict, dataclass
from typing import Iterable, List, Sequence

import torch

from src.utils import assert_finite_number, assert_finite_tensor


@dataclass
class QPResult:
    lambda_star: float
    active: bool
    skipped: bool
    skip_reason: str
    grad_ssl_norm: float
    grad_temporal_norm: float
    grad_dot: float
    grad_cosine: float
    a_dot_d0: float
    b: float
    correction_ratio: float

    def asdict(self):
        return asdict(self)


def grads_or_zeros(loss: torch.Tensor, params: Sequence[torch.nn.Parameter]) -> List[torch.Tensor]:
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    out = []
    for param, grad in zip(params, grads):
        out.append(torch.zeros_like(param) if grad is None else grad)
    return out


def dot(xs: Sequence[torch.Tensor], ys: Sequence[torch.Tensor]) -> torch.Tensor:
    total = None
    for x, y in zip(xs, ys):
        value = torch.sum(x * y)
        total = value if total is None else total + value
    if total is None:
        return torch.tensor(0.0)
    return total


def norm2(xs: Sequence[torch.Tensor]) -> torch.Tensor:
    return dot(xs, xs)


def solve_feasible_step(
    params: Sequence[torch.nn.Parameter],
    g_ssl: Sequence[torch.Tensor],
    g_temporal: Sequence[torch.Tensor],
    lr: float,
    weight_decay: float,
    r_value: float,
    rho: float,
    barrier_kappa: float,
    delta: float,
    zero_tol: float = 1e-12,
) -> tuple[List[torch.Tensor], QPResult]:
    d0 = [-(lr * (g + weight_decay * p.detach())) for p, g in zip(params, g_ssl)]
    a_norm2 = norm2(g_temporal)
    ssl_norm2 = norm2(g_ssl)
    grad_dot = dot(g_ssl, g_temporal)
    a_dot_d0 = dot(g_temporal, d0)
    b = float(barrier_kappa * (rho - r_value))

    for name, tensors in (("g_ssl", g_ssl), ("g_temporal", g_temporal), ("d0", d0)):
        for tensor in tensors:
            assert_finite_tensor(name, tensor)
    assert_finite_tensor("a_norm2", a_norm2)
    assert_finite_tensor("a_dot_d0", a_dot_d0)

    a_norm2_value = float(a_norm2.detach().cpu())
    ssl_norm = float(torch.sqrt(ssl_norm2.detach().clamp_min(0)).cpu())
    temporal_norm = float(torch.sqrt(a_norm2.detach().clamp_min(0)).cpu())
    grad_dot_value = float(grad_dot.detach().cpu())
    grad_cosine = grad_dot_value / (ssl_norm * temporal_norm + 1e-12)
    a_dot_d0_value = float(a_dot_d0.detach().cpu())

    if a_norm2_value <= zero_tol and a_dot_d0_value <= b:
        result = QPResult(
            lambda_star=0.0,
            active=False,
            skipped=False,
            skip_reason="",
            grad_ssl_norm=ssl_norm,
            grad_temporal_norm=temporal_norm,
            grad_dot=grad_dot_value,
            grad_cosine=grad_cosine,
            a_dot_d0=a_dot_d0_value,
            b=b,
            correction_ratio=0.0,
        )
        return d0, result

    if a_norm2_value <= zero_tol and r_value > rho:
        result = QPResult(
            lambda_star=0.0,
            active=False,
            skipped=True,
            skip_reason="temporal_gradient_near_zero_while_budget_violated",
            grad_ssl_norm=ssl_norm,
            grad_temporal_norm=temporal_norm,
            grad_dot=grad_dot_value,
            grad_cosine=grad_cosine,
            a_dot_d0=a_dot_d0_value,
            b=b,
            correction_ratio=0.0,
        )
        return [torch.zeros_like(p) for p in params], result

    lambda_star = max(0.0, (a_dot_d0_value - b) / (a_norm2_value + delta))
    updates = [d - lambda_star * a for d, a in zip(d0, g_temporal)]
    correction = [u - d for u, d in zip(updates, d0)]
    d0_norm = float(torch.sqrt(norm2(d0).detach().clamp_min(0)).cpu())
    correction_norm = float(torch.sqrt(norm2(correction).detach().clamp_min(0)).cpu())
    correction_ratio = correction_norm / (d0_norm + 1e-12)
    active = lambda_star > 0.0

    assert_finite_number("lambda_star", lambda_star)
    for update in updates:
        assert_finite_tensor("qp_update", update)

    result = QPResult(
        lambda_star=lambda_star,
        active=active,
        skipped=False,
        skip_reason="",
        grad_ssl_norm=ssl_norm,
        grad_temporal_norm=temporal_norm,
        grad_dot=grad_dot_value,
        grad_cosine=grad_cosine,
        a_dot_d0=a_dot_d0_value,
        b=b,
        correction_ratio=correction_ratio,
    )
    return updates, result


@torch.no_grad()
def apply_updates(params: Sequence[torch.nn.Parameter], updates: Sequence[torch.Tensor]) -> None:
    for param, update in zip(params, updates):
        param.add_(update)

