"""
Training utilities for supervised image classification.

This module contains the core training loop and keeps logging or hyperparameter
search concerns outside of the training code itself.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

EpochEndCallback = Callable[[dict[str, float | int | None]], None]


def _to_device(device: torch.device | str) -> torch.device:
    """Convert a device specification to ``torch.device``."""
    if isinstance(device, torch.device):
        return device
    return torch.device(device)


def _validate_num_epochs(num_epochs: int) -> None:
    """Validate the number of training epochs."""
    if not isinstance(num_epochs, int):
        msg = "num_epochs must be an integer."
        raise TypeError(msg)

    if num_epochs <= 0:
        msg = "num_epochs must be greater than 0."
        raise ValueError(msg)


def _get_current_lr(optimizer: torch.optim.Optimizer) -> float | None:
    """Return the learning rate of the first parameter group."""
    if len(optimizer.param_groups) == 0:
        return None
    return optimizer.param_groups[0].get("lr")


def _compute_epoch_metrics(
    *,
    total_loss: float,
    total_correct: int,
    total_samples: int,
) -> dict[str, float]:
    """Compute average loss and accuracy for one epoch."""
    if total_samples == 0:
        msg = "The dataloader produced zero samples for this epoch."
        raise ValueError(msg)

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }


def _clone_state_dict_to_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """
    Clone a model state dict to CPU.

    This avoids keeping an additional full copy of the best model on GPU memory.

    Parameters
    ----------
    model
        Model whose state dict should be cloned.

    Returns
    -------
    dict[str, torch.Tensor]
        Cloned state dict stored on CPU.

    """
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def train_one_epoch(
    *,
    model: torch.nn.Module,
    train_loader: Any,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
) -> dict[str, float]:
    """
    Train the model for one epoch.

    Parameters
    ----------
    model
        Model to train.
    train_loader
        Training dataloader yielding ``(images, targets)``.
    criterion
        Loss function.
    optimizer
        Optimizer used for parameter updates.
    device
        Target device.

    Returns
    -------
    dict[str, float]
        Dictionary with ``loss`` and ``accuracy``.

    """
    device = _to_device(device)
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_images, batch_targets in train_loader:
        images = batch_images.to(device, non_blocking=True)
        targets = batch_targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        preds = logits.argmax(dim=1)

        total_loss += loss.item() * batch_size
        total_correct += (preds == targets).sum().item()
        total_samples += batch_size

    return _compute_epoch_metrics(
        total_loss=total_loss,
        total_correct=total_correct,
        total_samples=total_samples,
    )


@torch.inference_mode()
def validate_one_epoch(
    *,
    model: torch.nn.Module,
    val_loader: Any,
    criterion: torch.nn.Module,
    device: torch.device | str,
) -> dict[str, float]:
    """
    Evaluate the model for one epoch.

    Parameters
    ----------
    model
        Model to evaluate.
    val_loader
        Validation dataloader yielding ``(images, targets)``.
    criterion
        Loss function.
    device
        Target device.

    Returns
    -------
    dict[str, float]
        Dictionary with ``loss`` and ``accuracy``.

    """
    device = _to_device(device)
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_images, batch_targets in val_loader:
        images = batch_images.to(device, non_blocking=True)
        targets = batch_targets.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, targets)

        batch_size = targets.size(0)
        preds = logits.argmax(dim=1)

        total_loss += loss.item() * batch_size
        total_correct += (preds == targets).sum().item()
        total_samples += batch_size

    return _compute_epoch_metrics(
        total_loss=total_loss,
        total_correct=total_correct,
        total_samples=total_samples,
    )


def _step_scheduler(
    scheduler: Any,
    *,
    val_loss: float,
) -> None:
    """Step the scheduler after each epoch."""
    if scheduler is None:
        return

    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(val_loss)
    else:
        scheduler.step()


def _should_update_best(
    *,
    current_val_accuracy: float,
    current_val_loss: float,
    best_val_accuracy: float,
    best_val_loss: float,
) -> bool:
    """Decide whether the current epoch is the new best epoch."""
    if current_val_accuracy > best_val_accuracy:
        return True

    return bool(current_val_accuracy == best_val_accuracy and current_val_loss < best_val_loss)


def train_model(
    *,
    model: torch.nn.Module,
    train_loader: Any,
    val_loader: Any,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
    num_epochs: int,
    scheduler: Any = None,
    epoch_end_callback: EpochEndCallback | None = None,
) -> dict[str, Any]:
    """
    Train a model for multiple epochs.

    Parameters
    ----------
    model
        Model to train.
    train_loader
        Training dataloader.
    val_loader
        Validation dataloader.
    criterion
        Loss function.
    optimizer
        Optimizer used for training.
    device
        Target device.
    num_epochs
        Number of training epochs.
    scheduler
        Optional learning-rate scheduler.
    epoch_end_callback
        Optional callback called after each epoch. It receives a dictionary
        with epoch metrics and may raise an exception, for example for pruning.

    Returns
    -------
    dict[str, Any]
        Dictionary containing the trained model, history, best metrics, and
        the best state dict.

    """
    _validate_num_epochs(num_epochs)

    device = _to_device(device)
    model = model.to(device)

    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "lr": [],
    }

    best_val_accuracy = float("-inf")
    best_val_loss = float("inf")
    best_epoch = -1
    best_state_dict = _clone_state_dict_to_cpu(model)

    for epoch in range(num_epochs):
        train_metrics = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_metrics = validate_one_epoch(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
        )

        current_lr = _get_current_lr(optimizer)

        history["train_loss"].append(train_metrics["loss"])
        history["train_accuracy"].append(train_metrics["accuracy"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["lr"].append(float(current_lr) if current_lr is not None else float("nan"))

        if _should_update_best(
            current_val_accuracy=val_metrics["accuracy"],
            current_val_loss=val_metrics["loss"],
            best_val_accuracy=best_val_accuracy,
            best_val_loss=best_val_loss,
        ):
            best_val_accuracy = val_metrics["accuracy"]
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch + 1
            best_state_dict = _clone_state_dict_to_cpu(model)

        epoch_info: dict[str, float | int | None] = {
            "epoch": epoch + 1,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "best_val_accuracy": best_val_accuracy,
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "lr": current_lr,
        }

        if epoch_end_callback is not None:
            epoch_end_callback(epoch_info)

        _step_scheduler(
            scheduler,
            val_loss=val_metrics["loss"],
        )

        print(
            f"Epoch {epoch + 1:03d}/{num_epochs:03d} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"train_acc={train_metrics['accuracy']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f}"
        )

    model.load_state_dict(best_state_dict)

    return {
        "model": model,
        "history": history,
        "best_val_accuracy": best_val_accuracy,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "best_state_dict": best_state_dict,
    }
