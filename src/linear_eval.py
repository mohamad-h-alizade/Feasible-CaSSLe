import time
from pathlib import Path
from typing import Dict, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data import build_eval_datasets, class_subset, limit_subset
from src.utils import append_csv, progress_interval, progress_print, save_json


def _label_map(classes, device):
    mapping = torch.full((100,), -1, dtype=torch.long, device=device)
    for idx, cls in enumerate(classes):
        mapping[int(cls)] = idx
    return mapping


@torch.no_grad()
def _linear_eval_accuracy(backbone, classifier, loader, device, task_items, class_map):
    backbone.eval()
    classifier.eval()
    correct = 0
    total = 0
    task_correct = {idx: 0 for idx, _ in task_items}
    task_total = {idx: 0 for idx, _ in task_items}
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = classifier(backbone(x))
        pred = logits.argmax(dim=1)
        mapped_y = class_map[y]
        valid = mapped_y >= 0
        correct_mask = pred[valid].eq(mapped_y[valid])
        correct += int(correct_mask.sum())
        total += int(valid.sum())
        for idx, cls in task_items:
            cls_tensor = cls.to(device)
            mask = torch.isin(y, cls_tensor)
            if int(mask.sum()) == 0:
                continue
            mapped_task_y = class_map[y[mask]]
            task_correct[idx] += int(pred[mask].eq(mapped_task_y).sum())
            task_total[idx] += int(mask.sum())
    out = {"linear_top1": 100.0 * correct / max(total, 1)}
    for idx, _ in task_items:
        if task_total[idx]:
            out[f"linear_task{idx}"] = 100.0 * task_correct[idx] / task_total[idx]
    return out


def run_linear_eval(
    cfg: Dict,
    model,
    tasks: Sequence[torch.Tensor],
    device,
    out_dir: Path,
    tag: str,
    seen_task_idx: int = None,
    task_indices: Sequence[int] = None,
) -> Dict:
    lin_cfg = cfg.get("linear_eval", {})
    if not lin_cfg.get("enabled", False):
        return {}

    train_eval, test_eval = build_eval_datasets(cfg)
    if task_indices is not None:
        task_items = [(idx, tasks[idx]) for idx in task_indices]
    elif seen_task_idx is not None:
        task_items = [(idx, tasks[idx]) for idx in range(seen_task_idx + 1)]
    else:
        task_items = [(idx, task) for idx, task in enumerate(tasks)]
    eval_tasks = [task for _, task in task_items]
    eval_classes = torch.cat(eval_tasks).tolist()
    train_eval = class_subset(train_eval, eval_classes)
    test_eval = class_subset(test_eval, eval_classes)
    max_train = lin_cfg.get("max_train_examples")
    max_test = lin_cfg.get("max_test_examples")
    if max_train:
        train_eval = limit_subset(train_eval, int(max_train), seed=int(cfg["experiment"]["seed"]) + 60_000)
    if max_test:
        test_eval = limit_subset(test_eval, int(max_test), seed=int(cfg["experiment"]["seed"]) + 70_000)

    batch_size = int(lin_cfg.get("batch_size", 256))
    num_workers = int(cfg["data"].get("num_workers", 0))
    train_loader = DataLoader(
        train_eval,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    test_loader = DataLoader(
        test_eval,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    backbone = model.encoder
    backbone_requires_grad = [param.requires_grad for param in backbone.parameters()]
    for param in backbone.parameters():
        param.requires_grad_(False)
    classifier = nn.Linear(model.features_dim, len(eval_classes)).to(device)
    class_map = _label_map(eval_classes, device)
    optimizer = torch.optim.SGD(
        classifier.parameters(),
        lr=float(lin_cfg.get("lr", 1.0)),
        momentum=float(lin_cfg.get("momentum", 0.0)),
        weight_decay=float(lin_cfg.get("weight_decay", 0.0)),
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[int(v) for v in lin_cfg.get("lr_decay_steps", [60, 80])],
        gamma=0.1,
    )

    epochs = int(lin_cfg.get("epochs", 100))
    log_path = out_dir / "linear_eval_log.csv"
    start = time.time()
    progress_print(cfg, f"[linear:{tag}] start epochs={epochs} train={len(train_eval)} test={len(test_eval)}")
    for epoch in range(epochs):
        backbone.eval()
        classifier.train()
        total_loss = 0.0
        total = 0
        for x, y in train_loader:
            x = x.to(device)
            y = class_map[y.to(device)]
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                feats = backbone(x)
            loss = torch.nn.functional.cross_entropy(classifier(feats), y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * int(y.numel())
            total += int(y.numel())
        scheduler.step()
        should_log = epoch == epochs - 1 or epoch % int(lin_cfg.get("log_frequency", 10)) == 0
        if should_log:
            append_csv(
                log_path,
                {
                    "tag": tag,
                    "epoch": epoch,
                    "train_loss": total_loss / max(total, 1),
                    "lr": optimizer.param_groups[0]["lr"],
                },
            )
        if should_log or (epoch + 1) % progress_interval(cfg) == 0:
            progress_print(
                cfg,
                f"[linear:{tag}] epoch={epoch + 1}/{epochs} "
                f"loss={total_loss / max(total, 1):.4f} lr={optimizer.param_groups[0]['lr']:.4g}",
            )

    metrics = _linear_eval_accuracy(backbone, classifier, test_loader, device, task_items, class_map)
    metrics["evaluator"] = "linear"
    metrics["accuracy"] = metrics["linear_top1"]
    if seen_task_idx is not None:
        task_values = [metrics[f"linear_task{idx}"] for idx in range(seen_task_idx + 1)]
        metrics["task"] = seen_task_idx
        metrics["current_task_accuracy"] = metrics[f"linear_task{seen_task_idx}"]
        metrics["avg_seen_accuracy"] = sum(task_values) / max(len(task_values), 1)
        metrics["linear_seen_top1"] = metrics["linear_top1"]
    metrics.update({"tag": tag, "linear_eval_wall_time_s": time.time() - start})
    save_json(out_dir / f"{tag}_linear_eval.json", metrics)
    progress_print(
        cfg,
        f"[eval] evaluator=linear tag={tag} accuracy={metrics['accuracy']:.3f}% "
        f"current={metrics.get('current_task_accuracy', metrics['accuracy']):.3f}% "
        f"avg_seen={metrics.get('avg_seen_accuracy', metrics['accuracy']):.3f}%",
    )
    for param, requires_grad in zip(backbone.parameters(), backbone_requires_grad):
        param.requires_grad_(requires_grad)
    return metrics
