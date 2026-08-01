from copy import deepcopy
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn

from src.cassle_compat import barlow_loss_func, byol_loss_func, require_cassle_methods, simclr_loss_func


class CassleTemporalPredictor(nn.Module):
    """Original CaSSLe distillation predictor architecture.

    The existing distiller wrappers use Linear -> BatchNorm1d -> ReLU -> Linear
    with a configurable hidden dimension. This module keeps that behavior
    outside Lightning for the support/query PoC.
    """

    def __init__(self, dim: int, hidden_dim: int = 2048, norm: str = "batchnorm"):
        super().__init__()
        layers: List[nn.Module] = [nn.Linear(dim, hidden_dim)]
        if norm == "batchnorm":
            layers.append(nn.BatchNorm1d(hidden_dim))
        elif norm == "layernorm":
            layers.append(nn.LayerNorm(hidden_dim))
        elif norm not in {"none", None}:
            raise ValueError(f"Unsupported predictor norm: {norm}")
        layers.extend([nn.ReLU(inplace=True), nn.Linear(hidden_dim, dim)])
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


def build_model(cfg: Dict, tasks=None, task_idx: int = 0) -> nn.Module:
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    opt_cfg = cfg["optimization"]
    method = model_cfg["method"]
    methods = require_cassle_methods()
    kwargs = dict(
        encoder=model_cfg.get("encoder", "resnet18"),
        num_classes=100,
        cifar=True,
        zero_init_residual=bool(model_cfg.get("zero_init_residual", False)),
        max_epochs=int(train_cfg.get("epochs_per_task", 1)),
        batch_size=int(cfg["data"].get("query_batch_size", 128)),
        online_eval_batch_size=None,
        optimizer="sgd",
        lars=False,
        lr=float(opt_cfg.get("lr", 0.03)),
        weight_decay=float(opt_cfg.get("weight_decay", 0.0)),
        classifier_lr=float(opt_cfg.get("lr", 0.03)),
        exclude_bias_n_norm=False,
        accumulate_grad_batches=1,
        extra_optimizer_args={},
        scheduler="none",
        min_lr=0.0,
        warmup_start_lr=0.0,
        warmup_epochs=0,
        multicrop=False,
        num_crops=2,
        num_small_crops=0,
        tasks=tasks,
        num_tasks=int(train_cfg.get("num_tasks", 5)),
        split_strategy="class",
        disable_knn_eval=True,
        knn_k=int(cfg["evaluation"].get("knn_k", 20)),
    )
    kwargs.update(model_cfg.get(method, {}))
    model = methods[method](**kwargs)
    model.current_task_idx = task_idx
    return model


def projection_dim(model: nn.Module) -> int:
    if hasattr(model, "projector"):
        for module in reversed(list(model.projector.modules())):
            if isinstance(module, nn.Linear):
                return int(module.out_features)
    raise ValueError("Could not infer SSL projection dimension")


def build_temporal_predictor(cfg: Dict, dim: int) -> CassleTemporalPredictor:
    temporal = cfg["temporal"]
    return CassleTemporalPredictor(
        dim=dim,
        hidden_dim=int(temporal.get("predictor_hidden_dim", 2048)),
        norm=temporal.get("predictor_norm", "batchnorm"),
    )


def representation_parameters(model: nn.Module) -> List[nn.Parameter]:
    params: List[nn.Parameter] = list(model.encoder.parameters())
    if hasattr(model, "projector"):
        params.extend(model.projector.parameters())
    if hasattr(model, "predictor"):
        params.extend(model.predictor.parameters())
    return [p for p in params if p.requires_grad]


def forward_project(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    return model(x)["z"]


def ssl_loss(cfg: Dict, model: nn.Module, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    method = cfg["model"]["method"]
    out1 = model(x1)
    out2 = model(x2)
    if method == "simclr":
        return simclr_loss_func(out1["z"], out2["z"], temperature=float(cfg["model"]["simclr"]["temperature"]))
    if method == "barlow_twins":
        barlow_cfg = cfg["model"]["barlow_twins"]
        return barlow_loss_func(
            out1["z"],
            out2["z"],
            lamb=float(barlow_cfg.get("lamb", 5e-3)),
            scale_loss=float(barlow_cfg.get("scale_loss", 0.025)),
        )
    if method == "byol":
        with torch.no_grad():
            m1 = model.momentum_projector(model.momentum_encoder(x1))
            m2 = model.momentum_projector(model.momentum_encoder(x2))
        return byol_loss_func(out1["p"], m2) + byol_loss_func(out2["p"], m1)
    raise ValueError(f"Unsupported SSL method: {method}")


def update_momentum_targets(model: nn.Module, tau: float) -> None:
    if not hasattr(model, "momentum_pairs"):
        return
    for online, target in model.momentum_pairs:
        for online_p, target_p in zip(online.parameters(), target.parameters()):
            target_p.data.mul_(tau).add_(online_p.data, alpha=1.0 - tau)


def reset_momentum_targets(model: nn.Module) -> None:
    if hasattr(model, "momentum_pairs"):
        for online, target in model.momentum_pairs:
            for online_param, target_param in zip(online.parameters(), target.parameters()):
                target_param.data.copy_(online_param.data)
                target_param.requires_grad = False


def freeze_historical_model(model: nn.Module) -> nn.Module:
    historical = deepcopy(model)
    historical.eval()
    for param in historical.parameters():
        param.requires_grad_(False)
    return historical


def checkpoint_state(model: nn.Module, temporal_predictor: nn.Module = None) -> Dict:
    state = {"model_state": model.state_dict()}
    if temporal_predictor is not None:
        state["temporal_predictor_state"] = temporal_predictor.state_dict()
    return state


def load_model_state(model: nn.Module, checkpoint_path: str) -> Dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("model_state", checkpoint.get("state_dict", checkpoint))
    model.load_state_dict(state, strict=False)
    return checkpoint if isinstance(checkpoint, dict) else {"model_state": state}
