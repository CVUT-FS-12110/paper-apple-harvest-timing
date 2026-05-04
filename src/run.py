import os
import logging
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from production.ref_models import RegressorModel
from dataset import ValueDataset


# -----------------------
# Config
# -----------------------
num_epochs = 100
batch_size = 64
scale = 100.0
device = "cuda" if torch.cuda.is_available() else "cpu"

experiment_name = "dayout"
dataset_root = Path("../data/dataset_class")


backbones = [
    "resnet18",
    #"mobilenet_v3_large", 
    #"densenet121",
    #"resnet50",
    #"efficientnet_b2",
    #"convnext_tiny",
]


val_strings = [
    '2024_0', '2024_98', '2024_35', '2024_70', '2024_7', '2024_42', '2024_77', '2024_14', '2024_49',
    '2024_84', '2024_21', '2024_56', '2024_91', '2024_28', '2024_63', '2025_0', '2025_-31', '2025_-28', '2025_4',
    '2025_7', '2025_-24', '2025_-21', '2025_11', '2025_14', '2025_-17', '2025_-14', '2025_18', '2025_21', '2025_-10', '2025_-7', '2025_-3'
]


# -----------------------
# Helpers
# -----------------------
def setup_logger(log_path: Path, logger_name: str) -> logging.Logger:
    """
    Create an isolated logger writing to log_path (no global basicConfig reuse).
    This avoids the common bug where basicConfig only applies once.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # don't duplicate into root

    # Clear previous handlers if any (important when re-creating in loops)
    if logger.handlers:
        for h in list(logger.handlers):
            logger.removeHandler(h)
            h.close()

    log_path.parent.mkdir(parents=True, exist_ok=True)

    handler = logging.FileHandler(log_path, mode="w")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt="%(asctime)s\t%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def to_numpy_list(x: torch.Tensor):
    # robust for scalar/batch outputs
    return x.detach().cpu().view(-1).numpy().tolist()


def parse_val_string(vs: str):
    year, day = vs.split("_")
    return year, int(day)


def build_day_mapping(val_strings_list):
    """
    Build sorted list of (day_value, val_string), sorted by day_value first and then val_string.
    Also compute the global numeric center as midpoint between min and max day.
    """
    entries = []
    day_values = []

    for vs in val_strings_list:
        year, day = parse_val_string(vs)
        entries.append((day, vs))
        day_values.append(day)

    entries.sort(key=lambda x: (x[0], x[1]))

    min_day = min(day_values)
    max_day = max(day_values)
    center_day = (min_day + max_day) / 2.0

    return entries, center_day, min_day, max_day


def select_validation_string(test_string: str, sorted_entries, center_day: float) -> str:
    """
    Validation = third day-group from test in the direction of the global center_day.

    Direction:
      - if test_day < center_day -> move to larger day values
      - if test_day > center_day -> move to smaller day values
      - if test_day == center_day -> move to larger day values

    We move in the globally sorted list of day-groups.
    If there are fewer than 3 positions available in that direction, clamp to the farthest available.
    """
    sorted_val_strings = [vs for _, vs in sorted_entries]

    if test_string not in sorted_val_strings:
        raise ValueError(f"Unknown test_string: {test_string}")

    _, test_day = parse_val_string(test_string)
    test_idx = sorted_val_strings.index(test_string)
    n = len(sorted_val_strings)

    if test_day < center_day:
        val_idx = min(test_idx + 3, n - 1)
    elif test_day > center_day:
        val_idx = max(test_idx - 3, 0)
    else:
        val_idx = min(test_idx + 3, n - 1)

    val_string = sorted_val_strings[val_idx]

    if val_string == test_string:
        raise RuntimeError(f"Validation day equals test day for {test_string}, which should never happen.")

    return val_string


# -----------------------
# Main experiment
# -----------------------
def main():
    # Ensure dirs exist
    Path("logs").mkdir(exist_ok=True)
    Path("checkpoints").mkdir(exist_ok=True)

    # Pre-collect file list (same for all runs)
    all_files = list(dataset_root.glob("*.jpg"))
    if len(all_files) == 0:
        raise FileNotFoundError(f"No .jpg files found in: {dataset_root.resolve()}")

    # Precompute global sorted day-groups and numeric center once
    sorted_entries, center_day, min_day, max_day = build_day_mapping(val_strings)

    for backbone in backbones:
        # One summary file per backbone
        summary_path = Path(f"logs/final_{experiment_name}_{backbone}.txt")
        # Fresh summary per backbone (overwrite)
        summary_path.write_text("")

        for val_string in val_strings:
            # --- Logging (per backbone + per val_string) ---
            log_path = Path(f"logs/detailed_logs/log_{experiment_name}_{backbone}_{val_string}.txt")
            logger = setup_logger(log_path, logger_name=f"{experiment_name}_{backbone}_{val_string}")

            val_split_string = select_validation_string(
                test_string=val_string,
                sorted_entries=sorted_entries,
                center_day=center_day
            )

            # --- Model ---
            model = RegressorModel(backbone=backbone, pretrained=True).to(device)

            # --- Collect data ---
            test_files = [f for f in all_files if val_string in str(f)]
            val_files = [f for f in all_files if val_split_string in str(f)]
            train_files = [f for f in all_files if (f not in test_files and f not in val_files)]

            if len(test_files) == 0:
                logger.info(f"WARNING: No test files matched val_string='{val_string}'. Skipping.")
                with open(summary_path, "a") as f:
                    f.write(f"Day {val_string}: SKIPPED (no test files)\n")
                continue

            if len(val_files) == 0:
                logger.info(f"WARNING: No validation files matched val_string='{val_split_string}'. Skipping.")
                with open(summary_path, "a") as f:
                    f.write(f"Day {val_string}: SKIPPED (no validation files for {val_split_string})\n")
                continue

            if len(train_files) == 0:
                logger.info(f"WARNING: No training files left after test='{val_string}' and val='{val_split_string}'. Skipping.")
                with open(summary_path, "a") as f:
                    f.write(f"Day {val_string}: SKIPPED (no training files)\n")
                continue

            # --- Datasets and dataloaders ---
            dataset_train = ValueDataset(train_files, augment=True)
            dataset_val = ValueDataset(val_files)
            dataset_test = ValueDataset(test_files)

            logger.info(
                f"Backbone: {backbone} | "
                f"Global center day: {center_day:.1f} (min={min_day}, max={max_day}) | "
                f"Test day: {val_string} | Validation day: {val_split_string} | "
                f"Training samples: {len(dataset_train)}, Validation samples: {len(dataset_val)}, Test samples: {len(dataset_test)}"
            )

            dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, pin_memory=True)
            dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, pin_memory=True)
            dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, pin_memory=True)

            # --- Optimizer and scheduler ---
            criterion = torch.nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

            best_mae = float("inf")
            best_test_mae = float("inf")

            # --- Training loop ---
            for epoch in range(num_epochs):
                # ---- TRAIN ----
                model.train()
                train_losses = []

                for inputs, targets in dataloader_train:
                    if inputs.ndim == 3:
                        inputs = inputs.unsqueeze(0)

                    inputs = inputs.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)

                    optimizer.zero_grad(set_to_none=True)
                    outputs = model(inputs)

                    targets_flat = targets.view(-1)
                    outputs_flat = outputs.view(-1)

                    loss = criterion(outputs_flat, targets_flat)
                    loss.backward()
                    optimizer.step()
                    train_losses.append(loss.item())

                # ---- VALIDATION ----
                model.eval()
                val_losses, val_outputs, val_targets = [], [], []
                with torch.no_grad():
                    for inputs, targets in dataloader_val:
                        if inputs.ndim == 3:
                            inputs = inputs.unsqueeze(0)

                        inputs = inputs.to(device, non_blocking=True)
                        targets = targets.to(device, non_blocking=True)

                        outputs = model(inputs)

                        targets_flat = targets.view(-1)
                        outputs_flat = outputs.view(-1)

                        loss = criterion(outputs_flat, targets_flat)
                        val_losses.append(loss.item())

                        val_outputs += to_numpy_list(outputs_flat)
                        val_targets += to_numpy_list(targets_flat)

                val_outputs = np.array(val_outputs, dtype=np.float32)
                val_targets = np.array(val_targets, dtype=np.float32)
                val_abs_errors = np.abs((val_outputs - val_targets) * scale)
                val_mae = float(np.mean(val_abs_errors))
                val_rmse = float(np.sqrt(np.mean(val_abs_errors ** 2)))
                val_std = float(np.std(val_abs_errors))

                # ---- TEST ----
                test_outputs, test_targets = [], []
                with torch.no_grad():
                    for inputs, targets in dataloader_test:
                        if inputs.ndim == 3:
                            inputs = inputs.unsqueeze(0)

                        inputs = inputs.to(device, non_blocking=True)
                        targets = targets.to(device, non_blocking=True)

                        outputs = model(inputs)

                        test_outputs += to_numpy_list(outputs.view(-1))
                        test_targets += to_numpy_list(targets.view(-1))

                test_outputs = np.array(test_outputs, dtype=np.float32)
                test_targets = np.array(test_targets, dtype=np.float32)
                test_abs_errors = np.abs((test_outputs - test_targets) * scale)
                test_mae = float(np.mean(test_abs_errors))
                test_rmse = float(np.sqrt(np.mean(test_abs_errors ** 2)))
                test_std = float(np.std(test_abs_errors))

                # ---- Logging ----
                logger.info(
                    f"Epoch [{epoch + 1:03d}] | "
                    f"Train loss: {np.mean(train_losses):.6f} | "
                    f"Val loss: {np.mean(val_losses):.6f} | "
                    f"Val MAE: {val_mae:.4f} | Val RMSE: {val_rmse:.4f} | Val STD: {val_std:.4f} | "
                    f"Test MAE: {test_mae:.4f} | Test RMSE: {test_rmse:.4f} | Test STD: {test_std:.4f}"
                )

                scheduler.step()

                # ---- Save best ----
                if val_mae < best_mae:
                    best_mae = val_mae
                    best_test_mae = test_mae
                    ckpt_path = Path(f"checkpoints/model_{experiment_name}_{backbone}_{val_string}.pt")
                    torch.save(model.state_dict(), ckpt_path)
                    logger.info(f"Model saved: {ckpt_path}")

            # --- Summary (per backbone) ---
            with open(summary_path, "a") as f:
                f.write(
                    f"Backbone {backbone} | Test day {val_string} | Val day {val_split_string}: "
                    f"Best Val MAE: {best_mae:.4f}, Best Test MAE: {best_test_mae:.4f}\n"
                )

            # Close file handlers to avoid too many open files in long runs
            for h in list(logger.handlers):
                h.close()
                logger.removeHandler(h)


if __name__ == "__main__":
    main()