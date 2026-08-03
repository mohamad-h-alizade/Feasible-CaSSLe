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
from src.plots import generate_run_figures
from src.qp import grads_or_zeros, solve_feasible_step
from src.temporal_losses import TemporalLossAdapter, normalized_reconstruction
from src.trainer import (
    _simclr_query_losses,
    make_batches,
    run_method,
    train_isolated_learning_fwt_baselines,
    train_task1,
)
from src.utils import choose_device, git_commit, make_run_dir, progress_print, save_json, set_seed


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
    progress_print(cfg, f"[run] dir={run_dir}")
    progress_print(cfg, f"[run] device={device}")
    return cfg, device, run_dir, tasks


def run_smoke(config_path: str) -> Dict:
    cfg, device, run_dir, tasks = prepare_run(config_path)
    try:
        methods = cfg["experiment"].get("methods", ["feasible_cassle"])
        progress_print(cfg, f"[run] methods={methods}")
        task1 = None if set(methods) == {"offline_ssl"} else train_task1(cfg, run_dir, device, tasks)
        learning_fwt = (
            {}
            if set(methods) == {"offline_ssl"}
            else train_isolated_learning_fwt_baselines(cfg, run_dir, device, tasks)
        )
        summaries = []
        for method in methods:
            summaries.append(run_method(cfg, run_dir, method, task1, device, tasks, learning_fwt))
        write_accuracy_summary(run_dir, summaries)
        write_diagnostics_summary(run_dir)
        figures = generate_figures(cfg, run_dir)
        payload = {"run_dir": str(run_dir), "summaries": summaries, "figures": figures}
        save_json(run_dir / "smoke_summary.json", payload)
        return payload
    except BaseException:
        if not list(run_dir.glob("*/*")):
            shutil.rmtree(run_dir, ignore_errors=True)
        raise


def run_pilot_selection(config_path: str, methods: List[str] = None) -> Dict:
    cfg, device, run_dir, tasks = prepare_run(config_path)
    try:
        selected = methods or cfg["experiment"].get(
            "notebook_default_methods",
            ["finetune", "standard_cassle", "crossfit_cassle", "feasible_cassle"],
        )
        progress_print(cfg, f"[run] methods={selected}")
        task1 = None if set(selected) == {"offline_ssl"} else train_task1(cfg, run_dir, device, tasks)
        learning_fwt = (
            {}
            if set(selected) == {"offline_ssl"}
            else train_isolated_learning_fwt_baselines(cfg, run_dir, device, tasks)
        )
        summaries = [run_method(cfg, run_dir, method, task1, device, tasks, learning_fwt) for method in selected]
        write_accuracy_summary(run_dir, summaries)
        write_diagnostics_summary(run_dir)
        figures = generate_figures(cfg, run_dir)
        payload = {"run_dir": str(run_dir), "summaries": summaries, "figures": figures}
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


def generate_figures(cfg: Dict, run_dir: Path) -> List[str]:
    try:
        figures = generate_run_figures(cfg, run_dir)
    except Exception as exc:
        save_json(run_dir / "figures_error.json", {"error": str(exc)})
        progress_print(cfg, f"[figures] skipped: {exc}")
        return []
    if figures:
        progress_print(cfg, f"[figures] wrote {len(figures)} files to {run_dir / 'figures'}")
    return figures


def write_accuracy_summary(run_dir: Path, summaries: List[Dict]) -> None:
    rows = []
    preferred = [
        "method",
        "task",
        "evaluator",
        "accuracy",
        "current_task_accuracy",
        "avg_seen_accuracy",
        "isolated_task_accuracy",
        "learning_forward_transfer_accuracy",
        "learning_fwt_baseline",
        "avg_forgetting_accuracy_drop",
    ]
    for summary in summaries:
        method = summary["method"]
        if method == "offline_ssl":
            row = {
                "method": method,
                "task": "offline",
                "current_task_accuracy": "",
                "avg_seen_accuracy": "",
                "avg_forgetting_accuracy_drop": "",
            }
            row.update(summary.get("eval", {}))
            rows.append(row)
            continue
        for eval_row in summary.get("eval_history", []):
            row = {"method": method, **eval_row}
            rows.append(row)
    if not rows:
        return
    fieldnames = []
    for key in preferred:
        if any(key in row for row in rows):
            fieldnames.append(key)
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with (run_dir / "accuracy_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _float_or_none(value):
    if value in ("", None):
        return None
    return float(value)


def write_diagnostics_summary(run_dir: Path) -> None:
    rows = []
    for log_path in sorted(run_dir.glob("*/train_log.csv")):
        method = log_path.parent.name
        with log_path.open() as f:
            records = list(csv.DictReader(f))
        if not records:
            continue
        numeric = {}
        for key in [
            "S_Q",
            "D_Q_raw",
            "R_Q",
            "grad_dot",
            "grad_cosine",
            "grad_ssl_norm",
            "grad_temporal_norm",
            "lambda_star",
            "correction_ratio",
            "wall_time_s",
            "peak_memory_mb",
        ]:
            values = [_float_or_none(row.get(key)) for row in records]
            values = [v for v in values if v is not None]
            if values:
                numeric[f"mean_{key}"] = sum(values) / len(values)
                numeric[f"max_{key}"] = max(values)
        active_values = [row.get("active") for row in records if row.get("active") not in ("", None)]
        skipped_values = [row.get("skipped") for row in records if row.get("skipped") not in ("", None)]
        conflict_values = [
            row.get("gradient_conflict") for row in records if row.get("gradient_conflict") not in ("", None)
        ]
        row = {
            "method": method,
            "steps": len(records),
            "active_steps": sum(v == "True" for v in active_values),
            "active_rate": sum(v == "True" for v in active_values) / max(len(active_values), 1),
            "conflict_steps": sum(v == "True" for v in conflict_values),
            "conflict_rate": sum(v == "True" for v in conflict_values) / max(len(conflict_values), 1),
            "skipped_steps": sum(v == "True" for v in skipped_values),
            **numeric,
        }
        rows.append(row)
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with (run_dir / "diagnostics_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    save_json(run_dir / "diagnostics_summary.json", {"methods": rows})


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
    parser.add_argument("--run", choices=["smoke", "pilot", "confirm", "inspect"], default="smoke")
    parser.add_argument("--methods", nargs="*", default=None)
    args = parser.parse_args()
    try:
        if args.run == "smoke":
            result = run_smoke(args.config)
        elif args.run in {"pilot", "confirm"}:
            result = run_pilot_selection(args.config, methods=args.methods)
        else:
            result = inspect_one_update(args.config)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    print(result["run_dir"])


if __name__ == "__main__":
    main()
