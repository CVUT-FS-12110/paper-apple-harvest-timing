import numpy as np
import torch

from training.anchors import gaussian_anchor_targets, decode_anchor_logits_to_scalar


def to_numpy_list(x: torch.Tensor):
    return x.detach().cpu().view(-1).numpy().tolist()


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
    anchor_centers,
    sigma_scaled: float,
):
    model.train()
    losses = []

    anchor_centers = anchor_centers.to(device)

    for inputs, targets in dataloader:
        if inputs.ndim == 3:
            inputs = inputs.unsqueeze(0)

        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).view(-1)

        soft_targets = gaussian_anchor_targets(
            targets=targets,
            anchor_centers=anchor_centers,
            sigma_scaled=sigma_scaled,
            normalize=True,
        )

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, soft_targets)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def evaluate_anchor_model(
    model,
    dataloader,
    criterion,
    device,
    anchor_centers,
    sigma_scaled: float,
    scale: float = 100.0,
):
    model.eval()

    anchor_centers = anchor_centers.to(device)

    losses = []
    pred_scaled = []
    target_scaled = []

    for inputs, targets in dataloader:
        if inputs.ndim == 3:
            inputs = inputs.unsqueeze(0)

        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).view(-1)

        soft_targets = gaussian_anchor_targets(
            targets=targets,
            anchor_centers=anchor_centers,
            sigma_scaled=sigma_scaled,
            normalize=True,
        )

        logits = model(inputs)
        loss = criterion(logits, soft_targets)
        losses.append(loss.item())

        preds = decode_anchor_logits_to_scalar(logits, anchor_centers)

        pred_scaled += to_numpy_list(preds)
        target_scaled += to_numpy_list(targets)

    pred_scaled = np.array(pred_scaled, dtype=np.float32)
    target_scaled = np.array(target_scaled, dtype=np.float32)

    abs_errors_days = np.abs((pred_scaled - target_scaled) * scale)
    mae = float(np.mean(abs_errors_days))
    rmse = float(np.sqrt(np.mean(abs_errors_days ** 2)))
    std = float(np.std(abs_errors_days))

    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "mae": mae,
        "rmse": rmse,
        "std": std,
        "pred_scaled": pred_scaled,
        "target_scaled": target_scaled,
    }