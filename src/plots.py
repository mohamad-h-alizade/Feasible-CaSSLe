from pathlib import Path
from typing import Dict, Iterable, List, Optional

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


def _plot_task_curves(frame: pd.DataFrame, path: Path) -> bool:
    if frame.empty or "task" not in frame.columns:
        return False
    task_columns = sorted(
        [column for column in frame.columns if column.startswith("linear_task")],
        key=lambda column: int(column.replace("linear_task", "")),
    )
    if not task_columns:
        return False
    data = frame.dropna(subset=["task"]).copy()
    if data.empty:
        return False

    import matplotlib.pyplot as plt

    methods = data["method"].dropna().unique()
    fig, axes = plt.subplots(len(methods), 1, figsize=(7.4, max(3.0, 2.4 * len(methods))), dpi=140, squeeze=False)
    for ax, (method, group) in zip(axes[:, 0], data.groupby("method", sort=False)):
        group = group.sort_values("task")
        for column in task_columns:
            series = pd.to_numeric(group[column], errors="coerce") if column in group.columns else None
            if series is not None and series.notna().any():
                ax.plot(group["task"], series, marker="o", linewidth=1.4, label=column)
        ax.set_title(f"Per-Task Accuracy: {method}")
        ax.set_xlabel("After Task")
        ax.set_ylabel("Accuracy (%)")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def _plot_task_heatmap(frame: pd.DataFrame, path: Path, method: Optional[str] = None) -> bool:
    if frame.empty or "task" not in frame.columns:
        return False
    task_columns = sorted(
        [column for column in frame.columns if column.startswith("linear_task")],
        key=lambda column: int(column.replace("linear_task", "")),
    )
    if not task_columns:
        return False
    data = frame.copy()
    if method is not None:
        data = data[data["method"] == method]
    data = data.dropna(subset=["task"]).sort_values(["method", "task"])
    if data.empty:
        return False

    import matplotlib.pyplot as plt

    values = data[task_columns].apply(pd.to_numeric, errors="coerce")
    if values.dropna(how="all").empty:
        return False
    labels = [f"{row.method} T{int(row.task)}" for row in data.itertuples()]
    fig_height = max(3.2, 0.34 * len(labels) + 1.4)
    fig, ax = plt.subplots(figsize=(7.2, fig_height), dpi=140)
    image = ax.imshow(values.to_numpy(), aspect="auto", cmap="viridis", vmin=0, vmax=100)
    ax.set_title("Task Accuracy Matrix")
    ax.set_xlabel("Evaluated Task")
    ax.set_ylabel("Training Stage")
    ax.set_xticks(range(len(task_columns)))
    ax.set_xticklabels([column.replace("linear_task", "T") for column in task_columns])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    fig.colorbar(image, ax=ax, label="Accuracy (%)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def _plot_scatter(frame: pd.DataFrame, x: str, y: str, title: str, path: Path) -> bool:
    if frame.empty or x not in frame.columns or y not in frame.columns:
        return False
    data = frame.dropna(subset=[x, y])
    if data.empty:
        return False

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.6, 4.2), dpi=140)
    for method, group in data.groupby("method", sort=False):
        ax.scatter(group[x], group[y], label=method, s=44)
        last = group.sort_values("task").tail(1)
        for row in last.itertuples():
            ax.annotate(str(method), (getattr(row, x), getattr(row, y)), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_title(title)
    ax.set_xlabel("Current-Task Accuracy (%)")
    ax.set_ylabel("Average Forgetting (points)")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def _read_train_logs(run_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(run_dir.glob("*/train_log.csv")):
        frame = pd.read_csv(path)
        if "method" not in frame.columns:
            frame["method"] = path.parent.name
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def _plot_train_lines(frame: pd.DataFrame, y: str, title: str, path: Path, method: Optional[str] = None) -> bool:
    if frame.empty or "step" not in frame.columns or y not in frame.columns:
        return False
    data = frame.copy()
    if method is not None:
        data = data[data["method"] == method]
    data = _to_numeric(data, ["task", "step", y])
    data = data.dropna(subset=["task", "step", y])
    if data.empty:
        return False

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.4, 4.2), dpi=140)
    for label, group in data.groupby(["method", "task"], sort=False):
        group = group.sort_values("step")
        display = f"{label[0]} T{int(label[1])}"
        ax.plot(group["step"], group[y], linewidth=1.2, label=display)
    ax.set_title(title)
    ax.set_xlabel("Step Within Task")
    ax.set_ylabel(y)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def _plot_budget(frame: pd.DataFrame, path: Path) -> bool:
    if frame.empty or "R_Q" not in frame.columns or "rho" not in frame.columns:
        return False
    data = _to_numeric(frame[frame["method"] == "feasible_cassle"].copy(), ["task", "step", "R_Q", "rho"])
    data = data.dropna(subset=["task", "step", "R_Q", "rho"])
    if data.empty:
        return False

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.4, 4.2), dpi=140)
    for task, group in data.groupby("task", sort=False):
        group = group.sort_values("step")
        ax.plot(group["step"], group["R_Q"], linewidth=1.2, label=f"T{int(task)} R_Q")
    rho = float(data["rho"].dropna().iloc[0])
    ax.axhline(rho, color="black", linestyle="--", linewidth=1.2, label=f"rho={rho:g}")
    ax.set_title("Feasible CaSSLe Budget Trace")
    ax.set_xlabel("Step Within Task")
    ax.set_ylabel("Normalized Reconstruction Error")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def _plot_active_by_task(frame: pd.DataFrame, path: Path) -> bool:
    if frame.empty or "active" not in frame.columns or "task" not in frame.columns:
        return False
    data = frame[frame["method"] == "feasible_cassle"].copy()
    if data.empty:
        return False
    data["active_value"] = data["active"].astype(str).eq("True").astype(float)
    grouped = data.groupby("task", sort=True)["active_value"].mean().reset_index()
    return _plot_bar(grouped, "task", "active_value", "Feasible Constraint Active Rate by Task", path)


def _plot_cosine_histogram(frame: pd.DataFrame, path: Path) -> bool:
    if frame.empty or "grad_cosine" not in frame.columns:
        return False
    data = _to_numeric(frame, ["grad_cosine"]).dropna(subset=["grad_cosine"])
    if data.empty:
        return False

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=140)
    for method, group in data.groupby("method", sort=False):
        ax.hist(group["grad_cosine"], bins=30, alpha=0.45, label=method)
    ax.axvline(0.0, color="black", linewidth=1.0)
    ax.set_title("Gradient Cosine Distribution")
    ax.set_xlabel("cos(g_ssl, g_temporal)")
    ax.set_ylabel("Steps")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
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
        for plotter, filename in [
            (_plot_task_curves, "per_task_accuracy_curves.png"),
            (_plot_task_heatmap, "task_accuracy_heatmap.png"),
        ]:
            path = figures_dir / filename
            if plotter(task_rows, path):
                written.append(str(path))
        path = figures_dir / "plasticity_forgetting_scatter.png"
        if _plot_scatter(
            final_rows,
            "current_task_accuracy",
            "avg_forgetting_accuracy_drop",
            "Plasticity vs Forgetting",
            path,
        ):
            written.append(str(path))

    diagnostics_path = run_dir / "diagnostics_summary.csv"
    if diagnostics_path.exists():
        diagnostics = pd.read_csv(diagnostics_path)
        diagnostics = _to_numeric(
            diagnostics,
            ["active_rate", "conflict_rate", "mean_grad_cosine", "mean_correction_ratio", "max_peak_memory_mb"],
        )
        for column, title, filename in [
            ("active_rate", "QP Constraint Active Rate", "active_rate.png"),
            ("conflict_rate", "Gradient Conflict Rate", "gradient_conflict_rate.png"),
            ("mean_grad_cosine", "Mean Gradient Cosine", "gradient_cosine.png"),
            ("mean_correction_ratio", "Mean Correction Ratio", "correction_ratio.png"),
        ]:
            path = figures_dir / filename
            if _plot_bar(diagnostics, "method", column, title, path):
                written.append(str(path))

    train_logs = _read_train_logs(run_dir)
    if not train_logs.empty:
        for y, title, filename in [
            ("S_Q", "SSL Query Loss", "ssl_loss_curves.png"),
            ("R_Q", "Normalized Reconstruction Error", "normalized_reconstruction_curves.png"),
        ]:
            path = figures_dir / filename
            if _plot_train_lines(train_logs, y, title, path):
                written.append(str(path))
        path = figures_dir / "feasible_budget_trace.png"
        if _plot_budget(train_logs, path):
            written.append(str(path))
        path = figures_dir / "feasible_active_rate_by_task.png"
        if _plot_active_by_task(train_logs, path):
            written.append(str(path))
        path = figures_dir / "gradient_cosine_distribution.png"
        if _plot_cosine_histogram(train_logs, path):
            written.append(str(path))

    return written
