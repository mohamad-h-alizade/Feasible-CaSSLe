# Feasible CaSSLe HOWTO

This guide covers only the new proof-of-concept code in `src/`, `config/`, and `main.ipynb`. The original CaSSLe scripts and `README.md` are unchanged.

## 1. Environment

Install the original repo dependencies first. In this workspace, the PoC also needs:

```bash
python -m pip install pytorch-lightning==1.9.5 lightning-bolts==0.7.0 torchmetrics==0.11.4 einops
```

Set the dataset path if you do not want the default `./data`:

```bash
export DATA_DIR=/path/to/data
```

The expected CIFAR-100 layout is compatible with torchvision:

```text
$DATA_DIR/cifar100/train/cifar-100-python/
$DATA_DIR/cifar100/val/cifar-100-python/
```

## 2. Smoke Run

Smoke is a correctness check, not a scientific experiment. It uses few examples, very few updates, capped k-NN evaluation, and 1 epoch of linear evaluation.

```bash
python -m src.experiments \
  --config config/feasible_cassle_cifar100_smoke.yaml \
  --run smoke
```

The command prints a timestamped run directory under `outputs/feasible_cassle/`.

## 3. Pilot Run

Pilot uses full CIFAR-100, 5 class-incremental tasks, SimCLR, 25 epochs per task, one seed, local logs, k-NN diagnostics, and paper-style linear evaluation.

```bash
python -m src.experiments \
  --config config/feasible_cassle_cifar100_pilot.yaml \
  --run pilot
```

By default it runs:

```text
offline_ssl
finetune
standard_cassle
crossfit_cassle
feasible_cassle
```

To run only selected methods:

```bash
python -m src.experiments \
  --config config/feasible_cassle_cifar100_pilot.yaml \
  --run pilot \
  --methods offline_ssl feasible_cassle standard_cassle
```

## 4. Confirmation Run

Confirmation is prepared for longer validation after pilot results look credible: 50 epochs per task and three seed values recorded in config. It does not automatically launch a sweep.

```bash
python -m src.experiments \
  --config config/feasible_cassle_cifar100_confirm.yaml \
  --run confirm
```

## 5. Baselines

- `offline_ssl`: trains SimCLR once on all CIFAR-100 classes jointly. This is the non-continual upper-bound-style reference.
- `finetune`: trains each new task with only the current-task SSL loss.
- `standard_cassle`: updates with `S_Q + gamma * D_Q_raw` using original CaSSLe symmetric contrastive temporal loss.
- `crossfit_cassle`: updates the temporal predictor on support data, freezes it, then applies the penalty loss on query data.
- `feasible_cassle`: takes the SSL step when it satisfies the normalized temporal budget, otherwise applies the QP correction.
- `compute_matched_finetune`: spends comparable support-side compute but updates the encoder only with SSL.

## 6. Linear Evaluation

The PoC trains a frozen-feature linear classifier directly from each final checkpoint. Pilot and confirm defaults mirror the original CaSSLe linear scripts where practical:

```yaml
epochs: 100
optimizer: SGD
lr: 1.0
weight_decay: 0.0
scheduler: step
lr_decay_steps: [60, 80]
batch_size: 256
```

Linear outputs are saved in each method folder:

```text
linear_eval_log.csv
<method>_task4_linear_eval.json
```

`offline_ssl` writes:

```text
offline_ssl_linear_eval.json
```

## 7. Logs

Each run directory contains:

```text
config.json
class_order.json
shared/task0.ckpt
<method>/train_log.csv
<method>/eval_log.csv
<method>/summary.json
```

Important accuracy fields:

- `current_task_accuracy`: linear or k-NN accuracy on the task just learned.
- `avg_seen_accuracy`: mean accuracy over all tasks seen so far.
- `avg_forgetting_accuracy_drop`: mean drop on previous tasks from their best earlier score.
- `linear_top1`: final all-class linear top-1 accuracy.
- `linear_taskX`: final linear top-1 accuracy on task `X` classes.
- `accuracy_summary.csv`: run-level table combining k-NN and linear metrics across methods.

Important Feasible diagnostics:

- `R_Q`: normalized temporal reconstruction error.
- `rho`: allowed normalized reconstruction budget.
- `active`: whether the QP constraint corrected the SSL step.
- `lambda_star`: scalar QP dual variable.
- `grad_cosine`: cosine between SSL and temporal gradients.
- `correction_ratio`: size of the QP correction relative to the ordinary SSL step.

## 8. Notebook

Use `main.ipynb` to inspect configs, run guarded commands, load result files, plot metrics, and make a go/no-go assessment. Expensive cells are disabled by default with:

```python
RUN_TRAINING = False
```
