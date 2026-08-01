from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, Subset

from src.cassle_compat import load_repo_module


@dataclass
class SupportQueryBatch:
    support_indices: torch.Tensor
    support_views: Tuple[torch.Tensor, torch.Tensor]
    support_targets: torch.Tensor
    query_indices: torch.Tensor
    query_views: Tuple[torch.Tensor, torch.Tensor]
    query_targets: torch.Tensor


class IndexedCIFAR100(torchvision.datasets.CIFAR100):
    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        return index, image, target


def cifar_root(data_cfg: Dict) -> Path:
    return Path(data_cfg["data_dir"]) / Path(data_cfg.get("cifar_root") or "cifar100")


def cifar_task_order(seed: int, num_classes: int = 100, num_tasks: int = 5) -> List[torch.Tensor]:
    if num_classes % num_tasks != 0:
        raise ValueError("num_classes must divide num_tasks")
    return list(torch.randperm(num_classes, generator=torch.Generator().manual_seed(seed)).chunk(num_tasks))


def build_pretrain_dataset(cfg: Dict, tasks: Sequence[torch.Tensor], task_idx: int) -> Dataset:
    pretrain = load_repo_module(
        "cassle/utils/pretrain_dataloader.py",
        "_feasible_cassle_pretrain_dataloader",
    )
    data_cfg = cfg["data"]
    aug_cfg = cfg["augmentations"]
    transform = pretrain.prepare_transform(data_cfg["dataset"], multicrop=False, **aug_cfg)
    task_transform = pretrain.prepare_n_crop_transform(transform, num_crops=2)
    train_dataset = IndexedCIFAR100(
        cifar_root(data_cfg),
        train=True,
        download=bool(data_cfg.get("download", True)),
        transform=task_transform,
    )
    task_dataset, _ = pretrain.split_dataset(
        train_dataset,
        task_idx=task_idx,
        num_tasks=cfg["training"]["num_tasks"],
        split_strategy="class",
        tasks=tasks,
    )
    limit = data_cfg.get("limit_examples_per_task")
    if limit:
        task_dataset = limit_subset(task_dataset, int(limit), seed=cfg["experiment"]["seed"] + task_idx)
    return task_dataset


def build_all_pretrain_dataset(cfg: Dict) -> Dataset:
    pretrain = load_repo_module(
        "cassle/utils/pretrain_dataloader.py",
        "_feasible_cassle_pretrain_dataloader_all",
    )
    data_cfg = cfg["data"]
    aug_cfg = cfg["augmentations"]
    transform = pretrain.prepare_transform(data_cfg["dataset"], multicrop=False, **aug_cfg)
    task_transform = pretrain.prepare_n_crop_transform(transform, num_crops=2)
    train_dataset = IndexedCIFAR100(
        cifar_root(data_cfg),
        train=True,
        download=bool(data_cfg.get("download", True)),
        transform=task_transform,
    )
    limit = cfg["data"].get("offline_limit_examples")
    if limit:
        train_dataset = limit_subset(
            train_dataset,
            int(limit),
            seed=int(cfg["experiment"]["seed"]) + 50_000,
        )
    return train_dataset


def build_pretrain_loader(cfg: Dict, dataset: Dataset, batch_size: int = None) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size or int(cfg["data"]["query_batch_size"]),
        shuffle=True,
        num_workers=int(cfg["data"].get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
        drop_last=bool(cfg["data"].get("drop_last", True)),
    )


def limit_subset(dataset: Dataset, limit: int, seed: int) -> Dataset:
    n = min(limit, len(dataset))
    perm = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed))[:n]
    return Subset(dataset, perm.tolist())


def build_support_query_loader(cfg: Dict, dataset: Dataset) -> DataLoader:
    data_cfg = cfg["data"]
    batch_size = int(data_cfg["support_batch_size"]) + int(data_cfg["query_batch_size"])
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(data_cfg.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
        drop_last=bool(data_cfg.get("drop_last", True)),
    )


def support_query_batches(cfg: Dict, loader: Iterable) -> Iterator[SupportQueryBatch]:
    support_n = int(cfg["data"]["support_batch_size"])
    query_n = int(cfg["data"]["query_batch_size"])
    for indices, views, targets in loader:
        if isinstance(views, torch.Tensor):
            raise ValueError("Expected two stochastic views from NCropAugmentation")
        if len(views) != 2:
            raise ValueError(f"Expected exactly two views, received {len(views)}")
        support_indices = indices[:support_n].view(-1)
        query_indices = indices[support_n : support_n + query_n].view(-1)
        if len(set(support_indices.tolist()).intersection(set(query_indices.tolist()))) != 0:
            raise AssertionError("Support and query minibatches are not disjoint")
        yield SupportQueryBatch(
            support_indices=support_indices,
            support_views=(views[0][:support_n], views[1][:support_n]),
            support_targets=targets[:support_n],
            query_indices=query_indices,
            query_views=(
                views[0][support_n : support_n + query_n],
                views[1][support_n : support_n + query_n],
            ),
            query_targets=targets[support_n : support_n + query_n],
        )


def move_batch(batch: SupportQueryBatch, device: torch.device) -> SupportQueryBatch:
    return SupportQueryBatch(
        support_indices=batch.support_indices.to(device),
        support_views=(batch.support_views[0].to(device), batch.support_views[1].to(device)),
        support_targets=batch.support_targets.to(device),
        query_indices=batch.query_indices.to(device),
        query_views=(batch.query_views[0].to(device), batch.query_views[1].to(device)),
        query_targets=batch.query_targets.to(device),
    )


def build_eval_datasets(cfg: Dict) -> Tuple[Dataset, Dataset]:
    data_cfg = cfg["data"]
    root = cifar_root(data_cfg)
    t_val = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ]
    )
    # Use deterministic inputs for k-NN memory and query sets. Stochastic train
    # crops made identical checkpoints report different task-0 accuracies.
    t_train = t_val
    train_dataset = torchvision.datasets.CIFAR100(
        root,
        train=True,
        download=bool(data_cfg.get("download", True)),
        transform=t_train,
    )
    val_dataset = torchvision.datasets.CIFAR100(
        root,
        train=False,
        download=bool(data_cfg.get("download", True)),
        transform=t_val,
    )
    return train_dataset, val_dataset


def class_subset(dataset: Dataset, classes: Sequence[int]) -> Dataset:
    targets = torch.as_tensor(dataset.targets)
    class_tensor = torch.as_tensor([int(c) for c in classes], dtype=targets.dtype)
    mask = torch.isin(targets, class_tensor)
    return Subset(dataset, mask.nonzero(as_tuple=False).view(-1).tolist())
