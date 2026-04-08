"""
Experiment runner with Weights & Biases integration.

This module builds a reproducible run configuration from the provided objects,
creates a W&B run, and executes the training loop. Optional Optuna trials can
reuse the same ``run_experiment`` function, receive epoch-wise reports, and
trigger pruning cleanly.
"""

from __future__ import annotations

import os
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import torch
import wandb
from dotenv import load_dotenv

from p01_cnn_icosimal.training import train_model

EpochInfo = dict[str, float | int | None]
EpochEndCallback = Callable[[EpochInfo], None]


def _to_device(device: torch.device | str) -> torch.device:
    """Convert a device specification to ``torch.device``."""
    if isinstance(device, torch.device):
        return device
    return torch.device(device)


def _set_seed(
    seed: int,
    *,
    deterministic: bool = True,
) -> None:
    """Set all relevant random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.default_rng(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.use_deterministic_algorithms(False)


def _bundle_get(data_bundle: Any, key: str) -> Any:
    """Read a field from either a dict-like or attribute-based data bundle."""
    if isinstance(data_bundle, dict):
        if key not in data_bundle:
            msg = f"data_bundle is missing key '{key}'."
            raise KeyError(msg)
        return data_bundle[key]

    if not hasattr(data_bundle, key):
        msg = f"data_bundle is missing attribute '{key}'."
        raise AttributeError(msg)

    return getattr(data_bundle, key)


def _is_config_value(value: Any) -> bool:
    """Return whether a value is safe to store in the run config."""
    if isinstance(value, (str, int, float, bool, type(None))):
        return True

    if isinstance(value, (Path, torch.device, torch.dtype)):
        return True

    if isinstance(value, (list, tuple)):
        return all(_is_config_value(v) for v in value)

    if isinstance(value, dict):
        return all(isinstance(k, str) and _is_config_value(v) for k, v in value.items())

    return False


def _normalize_for_config(value: Any) -> Any:
    """Normalize values so they can be safely logged to W&B."""
    if isinstance(value, (Path, torch.device, torch.dtype)):
        return str(value)

    if isinstance(value, tuple):
        return [_normalize_for_config(v) for v in value]

    if isinstance(value, list):
        return [_normalize_for_config(v) for v in value]

    if isinstance(value, dict):
        return {str(k): _normalize_for_config(v) for k, v in value.items()}

    return value


def _extract_public_state(obj: Any, exclude: set[str] | None = None) -> dict[str, Any]:
    """Extract simple public attributes from an object's ``__dict__``."""
    exclude = exclude or set()
    state = getattr(obj, "__dict__", {})
    extracted: dict[str, Any] = {}

    for key, value in state.items():
        if key.startswith("_"):
            continue

        if key in exclude:
            continue

        if callable(value):
            continue

        if isinstance(
            value,
            (
                torch.Tensor,
                torch.nn.Parameter,
                torch.nn.Module,
                torch.optim.Optimizer,
            ),
        ):
            continue

        if _is_config_value(value):
            extracted[key] = _normalize_for_config(value)

    return extracted


def _split_data_config(data_config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Split raw data config into meaningful nested groups."""
    grouped: dict[str, dict[str, Any]] = {
        "dataset": {},
        "dataloader": {},
        "transforms": {},
        "normalization": {},
        "other": {},
    }

    dataset_keys = {
        "data_root",
        "dataset_name",
        "image_size",
        "num_classes",
        "class_names",
        "num_train_samples",
        "num_val_samples",
        "train_dir",
        "val_dir",
    }

    dataloader_keys = {
        "batch_size",
        "num_workers",
        "pin_memory",
        "persistent_workers",
        "drop_last",
        "shuffle",
    }

    transform_keys = {
        "use_flip",
        "randaugment_num_ops",
        "randaugment_magnitude",
        "resize_mode",
        "interpolation",
    }

    normalization_keys = {
        "mean",
        "std",
    }

    for key, value in data_config.items():
        if key in dataset_keys:
            grouped["dataset"][key] = value
        elif key in dataloader_keys:
            grouped["dataloader"][key] = value
        elif key in transform_keys:
            grouped["transforms"][key] = value
        elif key in normalization_keys:
            grouped["normalization"][key] = value
        else:
            grouped["other"][key] = value

    return {group_name: group_values for group_name, group_values in grouped.items() if group_values}


def _extract_model_config(model: torch.nn.Module) -> dict[str, Any]:
    """Extract model metadata for the run config."""
    architecture: dict[str, Any] = {
        "model_name": model.__class__.__name__,
    }

    parameters: dict[str, Any] = {
        "num_parameters_total": sum(p.numel() for p in model.parameters()),
        "num_parameters_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }

    public_state: dict[str, Any] = {}
    if hasattr(model, "get_model_config") and callable(model.get_model_config):
        model_config = model.get_model_config()

        if not isinstance(model_config, dict):
            msg = "model.get_model_config() must return a dict."
            raise TypeError(msg)

        for key, value in model_config.items():
            if _is_config_value(value):
                public_state[key] = _normalize_for_config(value)
    else:
        public_state = _extract_public_state(model, exclude={"training"})

    return {
        "architecture": architecture,
        "parameters": parameters,
        "details": public_state,
    }


def _extract_optimizer_config(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    """Extract optimizer metadata for the run config."""
    config: dict[str, Any] = {
        "name": optimizer.__class__.__name__,
        "num_param_groups": len(optimizer.param_groups),
    }

    if len(optimizer.param_groups) == 0:
        return config

    first_group = optimizer.param_groups[0]

    for key, value in first_group.items():
        if key == "params":
            continue

        if _is_config_value(value):
            config[key] = _normalize_for_config(value)

    return config


def _extract_criterion_config(criterion: torch.nn.Module) -> dict[str, Any]:
    """Extract criterion metadata for the run config."""
    config: dict[str, Any] = {
        "name": criterion.__class__.__name__,
    }
    config.update(_extract_public_state(criterion))
    return config


def _extract_scheduler_config(scheduler: Any) -> dict[str, Any]:
    """Extract scheduler metadata for the run config."""
    if scheduler is None:
        return {
            "name": None,
        }

    config: dict[str, Any] = {
        "name": scheduler.__class__.__name__,
    }
    config.update(_extract_public_state(scheduler))
    return config


def _extract_data_config(data_bundle: Any) -> dict[str, Any]:
    """Extract data-related metadata from the data bundle."""
    raw_data_config = _bundle_get(data_bundle, "data_config")

    if not isinstance(raw_data_config, dict):
        msg = "data_bundle.data_config must be a dict."
        raise TypeError(msg)

    normalized_data_config: dict[str, Any] = {}
    for key, value in raw_data_config.items():
        if _is_config_value(value):
            normalized_data_config[key] = _normalize_for_config(value)

    class_names = _bundle_get(data_bundle, "class_names")
    train_loader = _bundle_get(data_bundle, "train_loader")
    val_loader = _bundle_get(data_bundle, "val_loader")

    if class_names is not None:
        normalized_data_config["num_classes"] = len(class_names)
        normalized_data_config["class_names"] = list(class_names)

    if hasattr(train_loader, "dataset"):
        normalized_data_config["num_train_samples"] = len(train_loader.dataset)

    if hasattr(val_loader, "dataset"):
        normalized_data_config["num_val_samples"] = len(val_loader.dataset)

    return _split_data_config(normalized_data_config)


def _validate_data_bundle(data_bundle: Any) -> None:
    """Validate that the data bundle exposes the required fields."""
    required_keys = ["train_loader", "val_loader", "class_names", "data_config"]

    for key in required_keys:
        _bundle_get(data_bundle, key)


def _validate_train_model_result(result: Any) -> None:
    """Validate the result returned by ``train_model``."""
    if not isinstance(result, dict):
        msg = "train_model(...) must return a dict."
        raise TypeError(msg)


def _validate_prune_metric(prune_metric: str) -> None:
    """Validate the prune metric name used for Optuna reporting."""
    if not isinstance(prune_metric, str):
        msg = "prune_metric must be a string."
        raise TypeError(msg)

    if not prune_metric.strip():
        msg = "prune_metric must not be empty."
        raise ValueError(msg)


def _load_wandb_settings() -> tuple[str, str | None, str | None]:
    """Load W&B settings from the local environment."""
    load_dotenv()

    project_name = os.getenv("WANDB_PROJECT", "cnn-icosimal")
    entity = os.getenv("WANDB_ENTITY")
    api_key = os.getenv("WANDB_API_KEY")

    return project_name, entity, api_key


def _build_auto_run_name(
    *,
    model: torch.nn.Module,
    trial: Any = None,
    run_name_suffix: str | None = None,
) -> str:
    """Build an automatic run name from the model name and optional suffix."""
    model_name = model.__class__.__name__
    name_parts = [model_name]

    if trial is not None and hasattr(trial, "number"):
        name_parts.append(f"trial_{trial.number}")

    if run_name_suffix:
        clean_suffix = run_name_suffix.strip()
        if clean_suffix:
            name_parts.append(clean_suffix)

    return "__".join(name_parts)


def _update_last_epoch_info(
    last_epoch_info: dict[str, float | int | None],
    epoch_info: EpochInfo,
) -> None:
    """Store the latest epoch information for later summary updates."""
    last_epoch_info.clear()
    last_epoch_info.update(epoch_info)


def _update_wandb_summary_from_result(wandb_run: Any, result: dict[str, Any]) -> None:
    """Write final best metrics from the training result to the W&B summary."""
    if "best_val_accuracy" in result:
        wandb_run.summary["best_val_accuracy"] = result["best_val_accuracy"]

    if "best_val_loss" in result:
        wandb_run.summary["best_val_loss"] = result["best_val_loss"]

    if "best_epoch" in result:
        wandb_run.summary["best_epoch"] = result["best_epoch"]


def _update_wandb_summary_from_epoch_info(wandb_run: Any, epoch_info: EpochInfo) -> None:
    """Write the latest known epoch information to the W&B summary."""
    summary_mapping = {
        "last_epoch": "epoch",
        "last_train_loss": "train_loss",
        "last_train_accuracy": "train_accuracy",
        "last_val_loss": "val_loss",
        "last_val_accuracy": "val_accuracy",
        "last_lr": "lr",
        "best_val_accuracy": "best_val_accuracy",
        "best_val_loss": "best_val_loss",
        "best_epoch": "best_epoch",
    }

    for summary_key, epoch_key in summary_mapping.items():
        value = epoch_info.get(epoch_key)
        if value is not None:
            wandb_run.summary[summary_key] = value


def _make_epoch_end_callback(
    *,
    wandb_run: Any,
    last_epoch_info: dict[str, float | int | None],
    trial: Any = None,
    prune_metric: str = "val_accuracy",
) -> EpochEndCallback:
    """
    Create a callback that logs to W&B and optionally reports to Optuna.

    Parameters
    ----------
    wandb_run
        Active W&B run.
    last_epoch_info
        Mutable dictionary that stores the latest epoch information.
    trial
        Optional Optuna trial.
    prune_metric
        Metric key from ``epoch_info`` used for Optuna reporting.

    Returns
    -------
    EpochEndCallback
        Callback used by ``train_model`` after each epoch.

    """

    def epoch_end_callback(epoch_info: EpochInfo) -> None:
        _update_last_epoch_info(last_epoch_info, epoch_info)

        epoch_value = epoch_info["epoch"]
        if epoch_value is None:
            msg = "epoch value cannot be None."
            raise ValueError(msg)

        log_payload = {
            "train_loss": epoch_info["train_loss"],
            "train_accuracy": epoch_info["train_accuracy"],
            "val_loss": epoch_info["val_loss"],
            "val_accuracy": epoch_info["val_accuracy"],
        }

        if epoch_info["lr"] is not None:
            log_payload["lr"] = epoch_info["lr"]

        wandb_run.log(log_payload, step=int(epoch_value), commit=True)

        if trial is None:
            return

        if prune_metric not in epoch_info:
            msg = f"prune_metric '{prune_metric}' not found in epoch_info."
            raise KeyError(msg)

        metric_value = epoch_info[prune_metric]
        if metric_value is None:
            msg = f"prune_metric '{prune_metric}' cannot be None."
            raise ValueError(msg)

        if not isinstance(metric_value, (int, float)):
            msg = f"prune_metric '{prune_metric}' must be numeric."
            raise TypeError(msg)

        trial.report(float(metric_value), step=int(epoch_value))

        if trial.should_prune():
            raise optuna.TrialPruned

    return epoch_end_callback


def build_run_config(
    *,
    data_bundle: Any,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    num_epochs: int,
    scheduler: Any = None,
    extra_config: dict[str, Any] | None = None,
    seed: int | None = None,
    deterministic: bool = True,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    """
    Build the full run configuration from the provided objects.

    Parameters
    ----------
    data_bundle
        Bundle containing ``train_loader``, ``val_loader``, ``class_names``,
        and ``data_config``.
    model
        Model used for training.
    optimizer
        Optimizer used for training.
    criterion
        Loss function.
    num_epochs
        Number of epochs for training.
    scheduler
        Optional learning-rate scheduler.
    extra_config
        Optional extra metadata to add to the run config.
    seed
        Optional random seed used for reproducibility.
    deterministic
        Whether deterministic torch execution is requested.
    device
        Optional training device.

    Returns
    -------
    dict[str, Any]
        Nested run configuration dictionary.

    """
    normalized_device = str(_to_device(device)) if device is not None else None

    config: dict[str, Any] = {
        "data": _extract_data_config(data_bundle),
        "model": _extract_model_config(model),
        "optimization": {
            "optimizer": _extract_optimizer_config(optimizer),
            "scheduler": _extract_scheduler_config(scheduler),
            "criterion": _extract_criterion_config(criterion),
        },
        "training": {
            "num_epochs": num_epochs,
            "seed": seed,
            "deterministic": deterministic,
        },
        "runtime": {
            "device": normalized_device,
            "cuda_available": torch.cuda.is_available(),
        },
    }

    if extra_config is not None:
        if not isinstance(extra_config, dict):
            msg = "extra_config must be a dict."
            raise TypeError(msg)
        config["extra"] = _normalize_for_config(extra_config)

    return config


def run_experiment(
    *,
    data_bundle: Any,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device | str,
    num_epochs: int,
    scheduler: Any = None,
    trial: Any = None,
    project_name: str | None = None,
    entity: str | None = None,
    run_name: str | None = None,
    run_name_suffix: str | None = None,
    tags: list[str] | None = None,
    notes: str | None = None,
    group: str | None = None,
    job_type: str | None = None,
    extra_config: dict[str, Any] | None = None,
    prune_metric: str = "val_accuracy",
    seed: int | None = None,
    deterministic: bool = True,
) -> dict[str, Any]:
    """
    Run one full experiment with W&B logging and optional Optuna reporting.

    Parameters
    ----------
    data_bundle
        Bundle containing ``train_loader``, ``val_loader``, ``class_names``,
        and ``data_config``.
    model
        Model to train.
    optimizer
        Optimizer used for training.
    criterion
        Loss function.
    device
        Target device such as ``"cuda"`` or ``"cpu"``.
    num_epochs
        Number of epochs for training.
    scheduler
        Optional learning-rate scheduler.
    trial
        Optional Optuna trial. If provided, the same function can be reused
        inside an Optuna objective.
    project_name
        Optional W&B project name. Falls back to ``WANDB_PROJECT`` or
        ``"cnn-icosimal"``.
    entity
        Optional W&B entity. Falls back to ``WANDB_ENTITY``.
    run_name
        Optional explicit W&B run name. If omitted, a name is built from the
        model name and the optional suffix.
    run_name_suffix
        Optional suffix appended to the automatically generated run name.
    tags
        Optional W&B tags.
    notes
        Optional W&B notes.
    group
        Optional W&B group.
    job_type
        Optional W&B job type.
    extra_config
        Optional extra metadata added to the run config.
    prune_metric
        Metric name from the epoch callback payload used for Optuna pruning.
    seed
        Optional random seed used for reproducibility.
    deterministic
        Whether deterministic torch execution is requested.

    Returns
    -------
    dict[str, Any]
        Training result dictionary extended with the run config and W&B run
        metadata.

    Raises
    ------
    KeyError
        If the data bundle is missing required keys or the pruning metric is
        missing in the epoch callback payload.
    AttributeError
        If the data bundle is missing required attributes.
    TypeError
        If invalid object types are returned or provided.
    ValueError
        If invalid values are provided for Optuna reporting.
    optuna.TrialPruned
        If the Optuna pruner decides to prune the current trial.
    Exception
        Any exception raised during training is re-raised after updating the
        W&B run summary.

    """
    _validate_data_bundle(data_bundle)

    if seed is not None:
        _set_seed(seed, deterministic=deterministic)

    if trial is not None:
        _validate_prune_metric(prune_metric)

    device = _to_device(device)
    model = model.to(device)

    env_project_name, env_entity, env_api_key = _load_wandb_settings()

    final_project_name = project_name or env_project_name
    final_entity = entity if entity is not None else env_entity

    if env_api_key:
        wandb.login(key=env_api_key)

    if run_name is None:
        run_name = _build_auto_run_name(
            model=model,
            trial=trial,
            run_name_suffix=run_name_suffix,
        )

    run_config = build_run_config(
        data_bundle=data_bundle,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        num_epochs=num_epochs,
        scheduler=scheduler,
        extra_config=extra_config,
        seed=seed,
        deterministic=deterministic,
        device=device,
    )

    wandb_run = wandb.init(
        project=final_project_name,
        entity=final_entity,
        name=run_name,
        tags=tags,
        notes=notes,
        group=group,
        job_type=job_type,
        config=run_config,
    )

    last_epoch_info: dict[str, float | int | None] = {}

    epoch_end_callback = _make_epoch_end_callback(
        wandb_run=wandb_run,
        last_epoch_info=last_epoch_info,
        trial=trial,
        prune_metric=prune_metric,
    )

    try:
        result = train_model(
            model=model,
            train_loader=_bundle_get(data_bundle, "train_loader"),
            val_loader=_bundle_get(data_bundle, "val_loader"),
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            num_epochs=num_epochs,
            scheduler=scheduler,
            epoch_end_callback=epoch_end_callback,
        )

        _validate_train_model_result(result)

    except optuna.TrialPruned:
        _update_wandb_summary_from_epoch_info(wandb_run, last_epoch_info)
        wandb_run.summary["status"] = "pruned"
        wandb_run.finish()
        raise

    except Exception as exc:
        _update_wandb_summary_from_epoch_info(wandb_run, last_epoch_info)
        wandb_run.summary["status"] = "failed"
        wandb_run.summary["exception_type"] = exc.__class__.__name__
        wandb_run.summary["exception_message"] = str(exc)
        wandb_run.finish()
        raise

    else:
        _update_wandb_summary_from_result(wandb_run, result)
        wandb_run.summary["status"] = "finished"
        wandb_run.finish()
        return result
