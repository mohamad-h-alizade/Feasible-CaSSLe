import time
from pathlib import Path
from typing import Dict, Iterator, List, Sequence

import torch
from torch import nn

from src.cassle_compat import simclr_loss_func
from src.calibration import CalibrationResult, calibrate_task, support_predictor_step
from src.data import (
    build_all_pretrain_dataset,
    build_eval_datasets,
    build_pretrain_dataset,
    build_pretrain_loader,
    build_support_query_loader,
    cifar_task_order,
    class_subset,
    limit_subset,
    move_batch,
    support_query_batches,
)
from src.linear_eval import run_linear_eval
from src.metrics import (
    encode_dataset,
    seen_classes,
    summarize_forgetting,
    task_classes,
    weighted_knn_accuracy,
)
from src.models import (
    build_model,
    build_temporal_predictor,
    checkpoint_state,
    freeze_historical_model,
    load_model_state,
    projection_dim,
    representation_parameters,
    update_momentum_targets,
)
from src.qp import apply_updates, grads_or_zeros, solve_feasible_step
from src.temporal_losses import TemporalLossAdapter, normalized_reconstruction
from src.utils import (
    append_csv,
    assert_finite_tensor,
    clone_parameters,
    eval_mode,
    frozen_params,
    peak_memory_mb,
    preserve_batchnorm_stats,
    progress_interval,
    progress_print,
    reset_peak_memory,
    save_json,
)


METHODS = {
    "offline_ssl",
    "finetune",
    "standard_cassle",
    "crossfit_cassle",
    "feasible_cassle",
    "compute_matched_finetune",
}


def make_batches(cfg: Dict, task_idx: int, tasks: Sequence[torch.Tensor]):
    dataset = build_pretrain_dataset(cfg, tasks, task_idx)
    loader = build_support_query_loader(cfg, dataset)
    if len(loader) == 0:
        raise RuntimeError(
            "Task dataset is too small for support_batch_size + query_batch_size with drop_last=true"
        )
    while True:
        for batch in support_query_batches(cfg, loader):
            yield batch


def _simclr_query_losses(
    cfg: Dict,
    model: nn.Module,
    historical: nn.Module,
    predictor: nn.Module,
    adapter: TemporalLossAdapter,
    views,
    calibration: CalibrationResult = None,
    freeze_predictor: bool = True,
):
    x1, x2 = views
    out1 = model(x1)
    out2 = model(x2)
    z1, z2 = out1["z"], out2["z"]
    ssl = simclr_loss_func(z1, z2, temperature=float(cfg["model"]["simclr"]["temperature"]))
    if historical is None or predictor is None:
        return ssl, None, None
    with torch.no_grad(), eval_mode(historical):
        frozen_z1 = historical(x1)["z"].detach()
        frozen_z2 = historical(x2)["z"].detach()
    if freeze_predictor:
        predictor_context = frozen_params(predictor)
        mode_context = eval_mode(predictor)
    else:
        predictor.train()
        predictor_context = torch.enable_grad()
        mode_context = torch.enable_grad()
    with mode_context, predictor_context:
        p1 = predictor(z1)
        p2 = predictor(z2)
        raw = adapter(p1, p2, frozen_z1, frozen_z2)
    norm = None
    if calibration is not None:
        norm = normalized_reconstruction(
            raw,
            calibration.d_good,
            calibration.d_chance,
            float(cfg["temporal"].get("epsilon_num", 1e-8)),
        )
    return ssl, raw, norm


def _sgd_updates(params, grads, lr: float, weight_decay: float):
    return [-(lr * (grad + weight_decay * param.detach())) for param, grad in zip(params, grads)]


def _clear_parameter_grads(modules: Sequence[nn.Module]) -> None:
    for module in modules:
        if module is None:
            continue
        for param in module.parameters():
            param.grad = None


def train_task1(cfg: Dict, run_dir: Path, device: torch.device, tasks) -> Path:
    configured = cfg["training"].get("task1_checkpoint")
    if configured:
        path = Path(configured).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Configured task1_checkpoint does not exist: {path}")
        return path
    ckpt_path = run_dir / "shared" / "task0.ckpt"
    if ckpt_path.exists():
        return ckpt_path

    model = build_model(cfg, tasks=tasks, task_idx=0).to(device)
    params = representation_parameters(model)
    lr = float(cfg["optimization"]["lr"])
    weight_decay = float(cfg["optimization"].get("weight_decay", 0.0))
    max_steps = int(cfg["training"].get("max_query_updates_per_task", 0) or 10**12)
    rows = []
    step = 0
    reset_peak_memory(device)
    progress_print(cfg, "[shared] train task0 ordinary SSL")
    for _ in range(int(cfg["training"].get("task1_epochs", cfg["training"]["epochs_per_task"]))):
        batches = make_batches(cfg, 0, tasks)
        steps_this_epoch = len(build_support_query_loader(cfg, build_pretrain_dataset(cfg, tasks, 0)))
        for _ in range(steps_this_epoch):
            batch = move_batch(next(batches), device)
            ssl, _, _ = _simclr_query_losses(cfg, model, None, None, None, batch.query_views)
            assert_finite_tensor("task1_ssl_loss", ssl)
            grads = grads_or_zeros(ssl, params)
            apply_updates(params, _sgd_updates(params, grads, lr, weight_decay))
            update_momentum_targets(model, tau=0.99)
            rows.append(
                {
                    "task": 0,
                    "step": step,
                    "method": "task1_ssl",
                    "S_Q": float(ssl.detach().cpu()),
                    "peak_memory_mb": peak_memory_mb(device),
                }
            )
            if step % progress_interval(cfg) == 0:
                progress_print(cfg, f"[shared] step={step} S_Q={float(ssl.detach().cpu()):.4f}")
            step += 1
            if step >= max_steps:
                break
        if step >= max_steps:
            break

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            **checkpoint_state(model),
            "task_idx": 0,
            "class_order": [task.tolist() for task in tasks],
            "config": cfg,
        },
        ckpt_path,
    )
    for row in rows:
        append_csv(run_dir / "shared" / "train_log.csv", row)
    progress_print(cfg, f"[shared] saved {ckpt_path}")
    return ckpt_path


def train_offline_ssl(cfg: Dict, run_dir: Path, device: torch.device, tasks) -> Dict:
    method_dir = run_dir / "offline_ssl"
    method_dir.mkdir(parents=True, exist_ok=False)
    model = build_model(cfg, tasks=tasks, task_idx=0).to(device)
    params = representation_parameters(model)
    lr = float(cfg["optimization"]["lr"])
    weight_decay = float(cfg["optimization"].get("weight_decay", 0.0))
    dataset = build_all_pretrain_dataset(cfg)
    loader = build_pretrain_loader(cfg, dataset)
    epochs = int(cfg["training"].get("offline_epochs", cfg["training"]["epochs_per_task"]))
    max_steps = int(cfg["training"].get("offline_max_query_updates", 0) or 10**12)
    step = 0
    reset_peak_memory(device)
    progress_print(
        cfg,
        f"[offline_ssl] train joint SSL epochs={epochs} "
        f"max_steps={max_steps if max_steps < 10**12 else 'all'}",
    )
    for _ in range(epochs):
        for _, views, _ in loader:
            x1, x2 = views[0].to(device), views[1].to(device)
            ssl, _, _ = _simclr_query_losses(cfg, model, None, None, None, (x1, x2))
            assert_finite_tensor("offline_ssl_loss", ssl)
            grads = grads_or_zeros(ssl, params)
            apply_updates(params, _sgd_updates(params, grads, lr, weight_decay))
            update_momentum_targets(model, tau=0.99)
            append_csv(
                method_dir / "train_log.csv",
                {
                    "step": step,
                    "method": "offline_ssl",
                    "S_Q": float(ssl.detach().cpu()),
                    "peak_memory_mb": peak_memory_mb(device),
                },
            )
            if step % progress_interval(cfg) == 0:
                progress_print(cfg, f"[offline_ssl] step={step} S_Q={float(ssl.detach().cpu()):.4f}")
            step += 1
            if step >= max_steps:
                break
        if step >= max_steps:
            break

    ckpt = method_dir / "offline_ssl.ckpt"
    torch.save(
        {
            **checkpoint_state(model),
            "task_idx": "offline",
            "method": "offline_ssl",
            "class_order": [task.tolist() for task in tasks],
            "config": cfg,
        },
        ckpt,
    )
    eval_result = evaluate_model(
        cfg,
        model,
        tasks,
        int(cfg["training"]["num_tasks"]) - 1,
        device,
        method_dir,
        tag="offline_ssl",
    )
    summary = {"method": "offline_ssl", "checkpoint": str(ckpt), "eval": eval_result}
    save_json(method_dir / "summary.json", summary)
    progress_print(cfg, "[offline_ssl] done")
    return summary


def run_method(
    cfg: Dict,
    run_dir: Path,
    method: str,
    task1_checkpoint: Path,
    device: torch.device,
    tasks,
) -> Dict:
    if method not in METHODS:
        raise ValueError(f"Unknown method {method}; choose from {sorted(METHODS)}")
    if method == "offline_ssl":
        return train_offline_ssl(cfg, run_dir, device, tasks)

    method_dir = run_dir / method
    method_dir.mkdir(parents=True, exist_ok=False)
    progress_print(cfg, f"[{method}] start")
    adapter = TemporalLossAdapter.from_config(cfg)
    model = build_model(cfg, tasks=tasks, task_idx=0).to(device)
    load_model_state(model, str(task1_checkpoint))
    eval_history: List[Dict[str, float]] = []
    eval_history.append(evaluate_model(cfg, model, tasks, 0, device, method_dir, tag=f"{method}_task0"))
    append_csv(method_dir / "eval_log.csv", {"method": method, **eval_history[-1]})

    prev_ckpt = task1_checkpoint
    for task_idx in range(1, int(cfg["training"]["num_tasks"])):
        progress_print(cfg, f"[{method}] task={task_idx} load previous checkpoint")
        load_model_state(model, str(prev_ckpt))
        model.current_task_idx = task_idx
        historical = freeze_historical_model(model).to(device)
        predictor = build_temporal_predictor(cfg, projection_dim(model)).to(device)
        predictor_optimizer = torch.optim.Adam(
            predictor.parameters(),
            lr=float(cfg["temporal"].get("predictor_lr", 0.001)),
        )
        batches = make_batches(cfg, task_idx, tasks)
        calibration = None
        if method in {"crossfit_cassle", "feasible_cassle", "compute_matched_finetune"}:
            progress_print(cfg, f"[{method}] task={task_idx} calibrate temporal predictor")
            calibration = calibrate_task(
                cfg,
                model,
                historical,
                predictor,
                predictor_optimizer,
                adapter,
                batches,
                device,
            )
            save_json(method_dir / f"task{task_idx}_calibration.json", calibration.asdict())
        elif method == "standard_cassle":
            calibration = CalibrationResult(0.0, 1.0, 1.0, 0.0, float(cfg["temporal"]["rho"]))

        train_task_increment(
            cfg,
            model,
            historical,
            predictor,
            predictor_optimizer,
            adapter,
            calibration,
            batches,
            method,
            method_dir,
            task_idx,
            device,
            tasks,
        )
        task_ckpt = method_dir / f"task{task_idx}.ckpt"
        torch.save(
            {
                **checkpoint_state(model, predictor),
                "task_idx": task_idx,
                "method": method,
                "class_order": [task.tolist() for task in tasks],
                "calibration": None if calibration is None else calibration.asdict(),
                "config": cfg,
            },
            task_ckpt,
        )
        prev_ckpt = task_ckpt
        eval_history.append(
            evaluate_model(cfg, model, tasks, task_idx, device, method_dir, tag=f"{method}_task{task_idx}")
        )
        eval_history[-1]["avg_forgetting"] = summarize_forgetting(eval_history, task_idx)
        eval_history[-1]["avg_forgetting_accuracy_drop"] = eval_history[-1]["avg_forgetting"]
        append_csv(method_dir / "eval_log.csv", {"method": method, **eval_history[-1]})
        progress_print(
            cfg,
            f"[{method}] task={task_idx} eval={eval_history[-1].get('evaluator', 'unknown')} "
            f"accuracy={eval_history[-1].get('accuracy', eval_history[-1]['current_task_accuracy']):.3f}% "
            f"current={eval_history[-1]['current_task_accuracy']:.3f}% "
            f"avg_seen={eval_history[-1]['avg_seen_accuracy']:.3f}% "
            f"forget={eval_history[-1]['avg_forgetting_accuracy_drop']:.3f}",
        )

    summary = {"method": method, "eval_history": eval_history}
    save_json(method_dir / "summary.json", summary)
    progress_print(cfg, f"[{method}] done")
    return summary


def train_task_increment(
    cfg: Dict,
    model: nn.Module,
    historical: nn.Module,
    predictor: nn.Module,
    predictor_optimizer: torch.optim.Optimizer,
    adapter: TemporalLossAdapter,
    calibration: CalibrationResult,
    batches: Iterator,
    method: str,
    method_dir: Path,
    task_idx: int,
    device: torch.device,
    tasks,
) -> None:
    params = representation_parameters(model)
    lr = float(cfg["optimization"]["lr"])
    weight_decay = float(cfg["optimization"].get("weight_decay", 0.0))
    gamma = float(cfg["temporal"].get("distill_gamma", 1.0))
    max_steps = int(cfg["training"].get("max_query_updates_per_task", 0) or 10**12)
    epochs = int(cfg["training"]["epochs_per_task"])
    task_dataset = build_pretrain_dataset(cfg, task_idx=task_idx, tasks=tasks)
    steps_per_epoch = len(build_support_query_loader(cfg, task_dataset))
    total_steps = min(max_steps, epochs * steps_per_epoch)
    early_steps = int(total_steps * float(cfg["temporal"].get("predictor_early_fraction", 0.10)))
    reset_peak_memory(device)
    progress_print(cfg, f"[{method}] task={task_idx} train steps={total_steps}")

    for step in range(total_steps):
        batch = move_batch(next(batches), device)
        start_time = time.time()
        support_updates = 0
        if method in {"crossfit_cassle", "feasible_cassle", "compute_matched_finetune"}:
            updates = int(cfg["temporal"].get("predictor_updates_early", 3)) if step < early_steps else int(
                cfg["temporal"].get("predictor_updates_per_step", 1)
            )
            for _ in range(updates):
                support_predictor_step(
                    model,
                    historical,
                    predictor,
                    predictor_optimizer,
                    adapter,
                    batch.support_views,
                )
                support_updates += 1
        elif method == "standard_cassle":
            support_updates = 0

        ssl, raw, norm = _simclr_query_losses(
            cfg,
            model,
            historical if method != "finetune" else None,
            predictor if method != "finetune" else None,
            adapter,
            batch.query_views,
            calibration if method == "feasible_cassle" else None,
            freeze_predictor=method != "standard_cassle",
        )
        assert_finite_tensor("S_Q", ssl)
        row = {
            "task": task_idx,
            "step": step,
            "method": method,
            "support_updates": support_updates,
            "S_Q": float(ssl.detach().cpu()),
            "D_Q_raw": "" if raw is None else float(raw.detach().cpu()),
            "R_Q": "" if norm is None else float(norm.detach().cpu()),
            "rho": "" if calibration is None else calibration.rho,
            "D_good": "" if calibration is None else calibration.d_good,
            "D_chance": "" if calibration is None else calibration.d_chance,
            "epsilon_raw": "" if calibration is None else calibration.raw_budget,
            "skipped": False,
            "skip_reason": "",
        }

        if method == "finetune" or method == "compute_matched_finetune":
            grads = grads_or_zeros(ssl, params)
            apply_updates(params, _sgd_updates(params, grads, lr, weight_decay))
        elif method == "standard_cassle":
            total = ssl + gamma * raw
            grads = grads_or_zeros(total, params)
            predictor_optimizer.zero_grad(set_to_none=True)
            raw.backward()
            predictor_optimizer.step()
            apply_updates(params, _sgd_updates(params, grads, lr, weight_decay))
            row["loss_total"] = float(total.detach().cpu())
        elif method == "crossfit_cassle":
            total = ssl + gamma * raw
            grads = grads_or_zeros(total, params)
            apply_updates(params, _sgd_updates(params, grads, lr, weight_decay))
            row["loss_total"] = float(total.detach().cpu())
        elif method == "feasible_cassle":
            g_ssl = grads_or_zeros(ssl, params)
            g_temporal = grads_or_zeros(norm, params)
            updates, qp = solve_feasible_step(
                params,
                g_ssl,
                g_temporal,
                lr=lr,
                weight_decay=weight_decay,
                r_value=float(norm.detach().cpu()),
                rho=calibration.rho,
                barrier_kappa=float(cfg["temporal"].get("barrier_kappa", 0.1)),
                delta=float(cfg["temporal"].get("qp_delta", 1e-12)),
            )
            row.update(qp.asdict())
            if not qp.skipped:
                apply_updates(params, updates)
            else:
                support_predictor_step(
                    model,
                    historical,
                    predictor,
                    predictor_optimizer,
                    adapter,
                    batch.support_views,
                )
        else:
            raise ValueError(method)

        update_momentum_targets(model, tau=0.99)
        _clear_parameter_grads([model, predictor])
        row["wall_time_s"] = time.time() - start_time
        row["peak_memory_mb"] = peak_memory_mb(device)
        append_csv(method_dir / "train_log.csv", row)
        if step % progress_interval(cfg) == 0 or step == total_steps - 1:
            msg = f"[{method}] task={task_idx} step={step + 1}/{total_steps} S_Q={row['S_Q']:.4f}"
            if row["D_Q_raw"] != "":
                msg += f" D={row['D_Q_raw']:.4f}"
            if row["R_Q"] != "":
                msg += (
                    f" R={row['R_Q']:.4f} active={row.get('active')} "
                    f"corr={row.get('correction_ratio'):.4f}"
                )
            progress_print(cfg, msg)


def evaluate_model(
    cfg: Dict,
    model: nn.Module,
    tasks,
    task_idx: int,
    device: torch.device,
    out_dir: Path,
    tag: str,
) -> Dict:
    method = cfg.get("evaluation", {}).get("method", "linear").lower()
    if method == "linear":
        return run_linear_eval(cfg, model, tasks, device, out_dir, tag=tag, seen_task_idx=task_idx)
    if method == "knn":
        return evaluate_after_task(cfg, model, tasks, task_idx, device)
    raise ValueError("evaluation.method must be either 'linear' or 'knn'")


def evaluate_after_task(cfg: Dict, model: nn.Module, tasks, task_idx: int, device: torch.device) -> Dict:
    progress_print(cfg, f"[eval:knn] task={task_idx} start")
    train_eval, test_eval = build_eval_datasets(cfg)
    batch_size = int(cfg["evaluation"].get("batch_size", 256))
    num_workers = int(cfg["data"].get("num_workers", 0))
    max_train = cfg["evaluation"].get("max_eval_train_examples")
    max_test = cfg["evaluation"].get("max_eval_test_examples")
    seen = seen_classes(tasks, task_idx)
    train_seen = class_subset(train_eval, seen)
    if max_train:
        train_seen = limit_subset(
            train_seen,
            int(max_train),
            seed=int(cfg["experiment"]["seed"]) + 10_000 + task_idx,
        )
    train_features, train_targets = encode_dataset(model, train_seen, device, batch_size, num_workers)
    result = {"task": task_idx}
    task_accs = []
    for idx in range(task_idx + 1):
        test_task = class_subset(test_eval, task_classes(tasks, idx))
        if max_test:
            test_task = limit_subset(
                test_task,
                int(max_test),
                seed=int(cfg["experiment"]["seed"]) + 20_000 + task_idx * 100 + idx,
            )
        test_features, test_targets = encode_dataset(model, test_task, device, batch_size, num_workers)
        acc = weighted_knn_accuracy(
            train_features,
            train_targets,
            test_features,
            test_targets,
            k=int(cfg["evaluation"].get("knn_k", 20)),
        )
        result[f"knn_task{idx}"] = acc
        task_accs.append(acc)
    test_seen = class_subset(test_eval, seen)
    if max_test:
        test_seen = limit_subset(
            test_seen,
            int(max_test),
            seed=int(cfg["experiment"]["seed"]) + 30_000 + task_idx,
        )
    test_seen_features, test_seen_targets = encode_dataset(model, test_seen, device, batch_size, num_workers)
    result["knn_seen"] = weighted_knn_accuracy(
        train_features,
        train_targets,
        test_seen_features,
        test_seen_targets,
        k=int(cfg["evaluation"].get("knn_k", 20)),
    )
    result["current_task_knn"] = result[f"knn_task{task_idx}"]
    result["avg_seen_knn"] = sum(task_accs) / max(len(task_accs), 1)
    result["evaluator"] = "knn"
    result["accuracy"] = result["knn_seen"]
    result["current_task_accuracy"] = result["current_task_knn"]
    result["avg_seen_accuracy"] = result["avg_seen_knn"]
    progress_print(
        cfg,
        f"[eval] evaluator=knn task={task_idx} accuracy={result['accuracy']:.3f}% "
        f"current={result['current_task_accuracy']:.3f}% "
        f"avg_seen={result['avg_seen_accuracy']:.3f}%",
    )
    return result
