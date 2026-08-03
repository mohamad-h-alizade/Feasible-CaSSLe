from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset


@torch.no_grad()
def encode_dataset(
    model,
    dataset: Dataset,
    device: torch.device,
    batch_size: int = 256,
    num_workers: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    features = []
    targets = []
    was_training = model.training
    model.eval()
    for batch in loader:
        if len(batch) == 3:
            _, x, y = batch
        else:
            x, y = batch
        x = x.to(device)
        out = model.base_forward(x)
        features.append(out["feats"].detach().cpu())
        targets.append(y.detach().cpu())
    model.train(was_training)
    return torch.cat(features), torch.cat(targets)


def weighted_knn_accuracy(
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    test_features: torch.Tensor,
    test_targets: torch.Tensor,
    k: int = 20,
    temperature: float = 0.07,
    max_distance_matrix_size: int = int(5e6),
) -> float:
    train_features = torch.nn.functional.normalize(train_features, dim=1)
    test_features = torch.nn.functional.normalize(test_features, dim=1)
    classes = torch.unique(torch.cat([train_targets, test_targets])).sort()[0]
    class_to_col = {int(c): i for i, c in enumerate(classes.tolist())}
    mapped_train = torch.tensor([class_to_col[int(c)] for c in train_targets.tolist()])
    mapped_test = torch.tensor([class_to_col[int(c)] for c in test_targets.tolist()])

    k = min(k, train_features.size(0))
    chunk_size = min(max(1, max_distance_matrix_size // train_features.size(0)), test_features.size(0))
    correct = 0
    total = 0
    for start in range(0, test_features.size(0), chunk_size):
        feats = test_features[start : start + chunk_size]
        targets = mapped_test[start : start + chunk_size]
        sim = torch.mm(feats, train_features.t())
        distances, indices = sim.topk(k, largest=True, sorted=True)
        neighbors = mapped_train[indices]
        weights = torch.exp(distances / temperature)
        probs = torch.zeros(feats.size(0), classes.numel())
        probs.scatter_add_(1, neighbors, weights)
        preds = probs.argmax(dim=1)
        correct += int((preds == targets).sum())
        total += int(targets.numel())
    return 100.0 * correct / max(total, 1)


def seen_classes(tasks: Sequence[torch.Tensor], task_idx: int) -> List[int]:
    return torch.cat([tasks[i] for i in range(task_idx + 1)]).tolist()


def task_classes(tasks: Sequence[torch.Tensor], task_idx: int) -> List[int]:
    return tasks[task_idx].tolist()


def summarize_forgetting(eval_history: List[Dict[str, float]], current_task: int) -> float:
    if current_task == 0:
        return 0.0
    forgetting = []
    current = eval_history[-1]
    evaluator = current.get("evaluator", "knn")
    prefix = "linear_task" if evaluator == "linear" else "knn_task"
    for task_idx in range(current_task):
        key = f"{prefix}{task_idx}"
        if key not in current:
            continue
        best_before = max(float(row.get(key, 0.0) or 0.0) for row in eval_history[:-1])
        forgetting.append(best_before - float(current.get(key, 0.0) or 0.0))
    return sum(forgetting) / max(len(forgetting), 1)
