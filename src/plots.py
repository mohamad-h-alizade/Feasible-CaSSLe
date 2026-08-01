from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

import matplotlib

matplotlib.use("Agg")


def _to_numeric(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    frame = frame.copy()
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _plot_lines(frame: pd.DataFrame, x: str, y: str, title: str, path: Path) -> bool:
    if frame.empty or x not in frame.columns or y not in frame.columns:
        return False
    data = frame.dropna(subset=[x, y])
    if data.empty:
        return False

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=140)
    for method, group in data.groupby("method", sort=False):
        group = group.sort_values(x)
        ax.plot(group[x], group[y], marker="o", linewidth=1.8, label=method)
    ax.set_title(title)
    ax.set_xlabel("Task")
    ax.set_ylabel("Accuracy (%)" if "accuracy" in y else y)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def _plot_bar(frame: pd.DataFrame, x: str, y: str, title: str, path: Path) -> bool:
    if frame.empty or x not in frame.columns or y not in frame.columns:
        return False
    data = frame.dropna(subset=[x, y])
    if data.empty:
        return False

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=140)
    ax.bar(data[x].astype(str), data[y])
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.tick_params(axis="x", rotation=25)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def generate_run_figures(cfg: Dict, run_dir: Path) -> List[str]:
    if not bool(cfg.get("plots", {}).get("enabled", True)):
        return []

    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    written: List[str] = []

    accuracy_path = run_dir / "accuracy_summary.csv"
    if accuracy_path.exists():
        accuracy = pd.read_csv(accuracy_path)
        accuracy = _to_numeric(
            accuracy,
            [
                "task",
                "accuracy",
                "current_task_accuracy",
                "avg_seen_accuracy",
                "avg_forgetting_accuracy_drop",
                "learning_forward_transfer_accuracy",
                "isolated_task_accuracy",
            ],
        )
        task_rows = accuracy[pd.to_numeric(accuracy.get("task"), errors="coerce").notna()].copy()
        for column, title, filename in [
            ("current_task_accuracy", "Current-Task Accuracy", "current_task_accuracy.png"),
            ("avg_seen_accuracy", "Average Seen Accuracy", "avg_seen_accuracy.png"),
            ("avg_forgetting_accuracy_drop", "Average Forgetting", "avg_forgetting.png"),
            (
                "learning_forward_transfer_accuracy",
                "Learning Forward Transfer",
                "learning_forward_transfer.png",
            ),
        ]:
            path = figures_dir / filename
            if _plot_lines(task_rows, "task", column, title, path):
                written.append(str(path))

        final_rows = task_rows.sort_values("task").groupby("method", sort=False).tail(1)
        path = figures_dir / "final_accuracy_by_method.png"
        if _plot_bar(final_rows, "method", "accuracy", "Final Selected-Evaluator Accuracy", path):
            written.append(str(path))

    diagnostics_path = run_dir / "diagnostics_summary.csv"
    if diagnostics_path.exists():
        diagnostics = pd.read_csv(diagnostics_path)
        diagnostics = _to_numeric(
            diagnostics,
            ["active_rate", "mean_grad_cosine", "mean_correction_ratio", "max_peak_memory_mb"],
        )
        for column, title, filename in [
            ("active_rate", "QP Constraint Active Rate", "active_rate.png"),
            ("mean_grad_cosine", "Mean Gradient Cosine", "gradient_cosine.png"),
            ("mean_correction_ratio", "Mean Correction Ratio", "correction_ratio.png"),
        ]:
            path = figures_dir / filename
            if _plot_bar(diagnostics, "method", column, title, path):
                written.append(str(path))

    return written
