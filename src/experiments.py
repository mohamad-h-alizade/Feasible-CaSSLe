import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Dict, List

import torch

from src.config import config_with_metadata, load_config
from src.cassle_compat import require_cassle_methods
from src.data import cifar_task_order
from src.models import build_model, build_temporal_predictor, freeze_historical_model, projection_dim
from src.models import representation_parameters
from src.qp import grads_or_zeros, solve_feasible_step
from src.temporal_losses import TemporalLossAdapter, normalized_reconstruction
from src.trainer import _simclr_query_losses, make_batches, run_method, train_task1
from src.utils import choose_device, git_commit, make_run_dir, save_json, set_seed


def prepare_run(config_path: str, overrides: Dict = None):
    cfg = load_config(config_path, overrides=overrides or {})
    require_cassle_methods()
    set_seed(int(cfg["experiment"]["seed"]))
    device = choose_device(cfg["experiment"].get("device", "auto"))
    run_dir = make_run_dir(cfg["experiment"]["output_dir"], cfg["experiment"]["name"])
    save_json(run_dir / "config.json", config_with_metadata(cfg, git_commit()))
    tasks = cifar_task_order(
        int(cfg["experiment"]["seed"]),
        num_classes=100,
        num_tasks=int(cfg["training"]["num_tasks"]),
    )
    save_json(run_dir / "class_order.json", {"tasks": [task.tolist() for task in tasks]})
    return cfg, device, run_dir, tasks


def run_smoke(config_path: str) -> Dict:
    cfg, device, run_dir, tasks = prepare_run(config_path)
    try:
        task1 = train_task1(cfg, run_dir, device, tasks)
        summaries = []
        for method in cfg["experiment"].get("methods", ["feasible_cassle"]):
            summaries.append(run_method(cfg, run_dir, method, task1, device, tasks))
        payload = {"run_dir": str(run_dir), "summaries": summaries}
        save_json(run_dir / "smoke_summary.json", payload)
        return payload
    except BaseException:
        if not list(run_dir.glob("*/*")):
            shutil.rmtree(run_dir, ignore_errors=True)
        raise


def run_pilot_selection(config_path: str, methods: List[str] = None) -> Dict:
    cfg, device, run_dir, tasks = prepare_run(config_path)
    try:
        task1 = train_task1(cfg, run_dir, device, tasks)
        selected = methods or cfg["experiment"].get(
            "notebook_default_methods",
            ["finetune", "standard_cassle", "crossfit_cassle", "feasible_cassle"],
        )
        summaries = [run_method(cfg, run_dir, method, task1, device, tasks) for method in selected]
        payload = {"run_dir": str(run_dir), "summaries": summaries}
        save_json(run_dir / "pilot_summary.json", payload)
        return payload
    except BaseException:
        if not list(run_dir.glob("*/*")):
            shutil.rmtree(run_dir, ignore_errors=True)
        raise


def load_results(run_dir: str) -> Dict:
    root = Path(run_dir)
    payload = {"run_dir": str(root), "eval_logs": {}, "train_logs": {}, "summaries": {}}
    for path in root.glob("*/eval_log.csv"):
        with path.open() as f:
            payload["eval_logs"][path.parent.name] = list(csv.DictReader(f))
    for path in root.glob("*/train_log.csv"):
        with path.open() as f:
            payload["train_logs"][path.parent.name] = list(csv.DictReader(f))
    for path in root.glob("*/summary.json"):
        with path.open() as f:
            payload["summaries"][path.parent.name] = json.load(f)
    return payload


def inspect_one_update(config_path: str) -> Dict:
    cfg, device, run_dir, tasks = prepare_run(config_path)
    model = build_model(cfg, tasks=tasks, task_idx=0).to(device)
    historical = freeze_historical_model(model).to(device)
    predictor = build_temporal_predictor(cfg, projection_dim(model)).to(device)
    adapter = TemporalLossAdapter.from_config(cfg)
    batch = next(make_batches(cfg, 1, tasks))
    from src.data import move_batch

    batch = move_batch(batch, device)
    ssl, raw, _ = _simclr_query_losses(
        cfg,
        model,
        historical,
        predictor,
        adapter,
        batch.query_views,
        calibration=None,
        freeze_predictor=True,
    )
    d_good = float(raw.detach().cpu()) * 0.95
    d_chance = max(float(raw.detach().cpu()) * 1.2, d_good + 1e-6)
    norm = normalized_reconstruction(raw, d_good, d_chance, float(cfg["temporal"].get("epsilon_num", 1e-8)))
    params = representation_parameters(model)
    g_ssl = grads_or_zeros(ssl, params)
    g_temporal = grads_or_zeros(norm, params)
    updates, qp = solve_feasible_step(
        params,
        g_ssl,
        g_temporal,
        lr=float(cfg["optimization"]["lr"]),
        weight_decay=float(cfg["optimization"].get("weight_decay", 0.0)),
        r_value=float(norm.detach().cpu()),
        rho=float(cfg["temporal"]["rho"]),
        barrier_kappa=float(cfg["temporal"].get("barrier_kappa", 0.1)),
        delta=float(cfg["temporal"].get("qp_delta", 1e-12)),
    )
    lhs_after = qp.a_dot_d0 - qp.lambda_star * (qp.grad_temporal_norm**2)
    payload = {
        "run_dir": str(run_dir),
        "S_Q": float(ssl.detach().cpu()),
        "D_Q_raw": float(raw.detach().cpu()),
        "R_Q": float(norm.detach().cpu()),
        "first_order_lhs_after": lhs_after,
        **qp.asdict(),
    }
    save_json(run_dir / "one_update_inspection.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run", choices=["smoke", "pilot", "inspect"], default="smoke")
    parser.add_argument("--methods", nargs="*", default=None)
    args = parser.parse_args()
    try:
        if args.run == "smoke":
            result = run_smoke(args.config)
        elif args.run == "pilot":
            result = run_pilot_selection(args.config, methods=args.methods)
        else:
            result = inspect_one_update(args.config)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    print(result["run_dir"])


if __name__ == "__main__":
    main()
