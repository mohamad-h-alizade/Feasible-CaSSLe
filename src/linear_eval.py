import time
from pathlib import Path
from typing import Dict, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data import build_eval_datasets, class_subset, limit_subset
from src.metrics import task_classes
from src.utils import append_csv, save_json


@torch.no_grad()
def _linear_eval_accuracy(backbone, classifier, loader, device, tasks):
    backbone.eval()
    classifier.eval()
    correct = 0
    total = 0
    task_correct = {idx: 0 for idx in range(len(tasks))}
    task_total = {idx: 0 for idx in range(len(tasks))}
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = classifier(backbone(x))
        pred = logits.argmax(dim=1)
        correct_mask = pred.eq(y)
        correct += int(correct_mask.sum())
        total += int(y.numel())
        for idx, cls in enumerate(tasks):
            cls_tensor = cls.to(device)
            mask = torch.isin(y, cls_tensor)
            task_correct[idx] += int(correct_mask[mask].sum())
            task_total[idx] += int(mask.sum())
    out = {"linear_top1": 100.0 * correct / max(total, 1)}
    for idx in range(len(tasks)):
        if task_total[idx]:
            out[f"linear_task{idx}"] = 100.0 * task_correct[idx] / task_total[idx]
    return out


def run_linear_eval(cfg: Dict, model, tasks: Sequence[torch.Tensor], device, out_dir: Path, tag: str) -> Dict:
    lin_cfg = cfg.get("linear_eval", {})
    if not lin_cfg.get("enabled", False):
        return {}

    train_eval, test_eval = build_eval_datasets(cfg)
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
    for param in backbone.parameters():
        param.requires_grad_(False)
    classifier = nn.Linear(model.features_dim, 100).to(device)
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
    for epoch in range(epochs):
        backbone.eval()
        classifier.train()
        total_loss = 0.0
        total = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                feats = backbone(x)
            loss = torch.nn.functional.cross_entropy(classifier(feats), y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * int(y.numel())
            total += int(y.numel())
        scheduler.step()
        if epoch == epochs - 1 or epoch % int(lin_cfg.get("log_frequency", 10)) == 0:
            append_csv(
                log_path,
                {
                    "tag": tag,
                    "epoch": epoch,
                    "train_loss": total_loss / max(total, 1),
                    "lr": optimizer.param_groups[0]["lr"],
                },
            )

    metrics = _linear_eval_accuracy(backbone, classifier, test_loader, device, tasks)
    metrics.update({"tag": tag, "linear_eval_wall_time_s": time.time() - start})
    save_json(out_dir / f"{tag}_linear_eval.json", metrics)
    return metrics
