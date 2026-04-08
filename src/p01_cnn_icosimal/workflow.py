"""
Workflow module for running and evaluating CNN experiments.

This module provides:
- set_reproducibility: configure random seeds and deterministic behavior
- run_experiment_block: run or load a single experiment with optional artifact reuse
- evaluate_experiment_block: load and display evaluation results for an experiment
- export_experiment_model: export a trained model to an artifact directory
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from IPython.display import display

from p01_cnn_icosimal import evaluation, experiment_runner

if TYPE_CHECKING:
    from collections.abc import Callable

    from p01_cnn_icosimal.datasets import DataBundle


def set_reproducibility(seed: int, deterministic: bool = True) -> None:
    """Set Python and PyTorch RNGs plus deterministic torch behaviour."""
    random.seed(seed)
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


def run_experiment_block(
    *,
    experiment_key: str,
    build_components_fn: Callable[
        [],
        tuple[
            torch.nn.Module,
            torch.nn.Module,
            torch.optim.Optimizer,
            torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau | None,
        ],
    ],
    data_bundle: DataBundle,
    device: torch.device,
    output_dir: Path,
    seed: int,
    deterministic: bool,
    tags: list[str] | None = None,
    notes: str | None = None,
    group: str | None = None,
    job_type: str | None = "train",
    num_epochs: int = 5,
    artifact_run_dir: str | Path | None = None,
) -> dict:
    """Run or load one experiment and optionally reuse a full exported artifact package."""
    set_reproducibility(seed, deterministic=deterministic)

    if artifact_run_dir is not None:
        artifact_run_dir = Path(artifact_run_dir)

        required_files = [
            artifact_run_dir / "best_model.pt",
            artifact_run_dir / "config.json",
            artifact_run_dir / "summary.json",
            artifact_run_dir / "history.json",
        ]

        if all(path.exists() for path in required_files):
            saved_experiment = evaluation.load_saved_experiment(artifact_run_dir)
            summary = saved_experiment["summary"]

            result = {
                "history": saved_experiment["history"],
                "best_epoch": summary["best_epoch"],
                "best_val_accuracy": summary["best_val_accuracy"],
                "best_val_loss": summary["best_val_loss"],
                "run_config": saved_experiment["config"],
                "run_name": experiment_key,
                "output_dir": str(artifact_run_dir),
                "loaded_existing": True,
                "loaded_from_artifact": True,
            }

            print("Experiment key    :", experiment_key)
            print("Loaded existing   :", result["loaded_existing"])
            print("Loaded artifact   :", result["loaded_from_artifact"])
            print("Run name          :", result["run_name"])
            print("Output dir        :", result["output_dir"])
            print("Best epoch        :", result["best_epoch"])
            print("Best val accuracy :", f"{result['best_val_accuracy']:.4f}")
            print("Best val loss     :", f"{result['best_val_loss']:.4f}")

            return result

    model, criterion, optimizer, scheduler = build_components_fn()

    result = experiment_runner.run_or_load_experiment(
        data_bundle=data_bundle,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        num_epochs=num_epochs,
        scheduler=scheduler,
        run_name_suffix=experiment_key,
        tags=tags,
        notes=notes,
        group=group,
        job_type=job_type,
        seed=seed,
        deterministic=deterministic,
        save_outputs=True,
        base_output_dir=output_dir,
    )

    result["loaded_from_artifact"] = False

    print("Experiment key    :", experiment_key)
    print("Loaded existing   :", result["loaded_existing"])
    print("Loaded artifact   :", result["loaded_from_artifact"])
    print("Run name          :", result["run_name"])
    print("Output dir        :", result["output_dir"])
    print("Best epoch        :", result["best_epoch"])
    print("Best val accuracy :", f"{result['best_val_accuracy']:.4f}")
    print("Best val loss     :", f"{result['best_val_loss']:.4f}")

    return result


def evaluate_experiment_block(
    *,
    experiment_key: str,
    experiment_results: dict[str, dict],
    build_components_fn: Callable[
        [],
        tuple[
            torch.nn.Module,
            torch.nn.Module,
            torch.optim.Optimizer,
            torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau | None,
        ],
    ],
    data_bundle: DataBundle,
    device: torch.device,
) -> None:
    """Load the saved best model for one experiment and display its evaluation widget."""
    result = experiment_results[experiment_key]
    model_dir = Path(result["output_dir"])

    saved_experiment = evaluation.load_saved_experiment(model_dir)

    model, _, _, _ = build_components_fn()
    model.load_state_dict(
        torch.load(
            model_dir / "best_model.pt",
            map_location=device,
        )
    )

    print("Experiment key    :", experiment_key)
    print("Run name          :", result["run_name"])
    print("Output dir        :", result["output_dir"])
    print("Loaded artifact   :", result.get("loaded_from_artifact", False))
    print("Best epoch        :", result["best_epoch"])
    print("Best val accuracy :", f"{result['best_val_accuracy']:.4f}")
    print("Best val loss     :", f"{result['best_val_loss']:.4f}")

    eval_widget = evaluation.create_evaluation_widget(
        result_or_saved=saved_experiment,
        model=model,
        dataloader=data_bundle.val_loader,
        device=device,
        data_bundle=data_bundle,
    )

    display(eval_widget)


def export_experiment_model(
    *,
    experiment_key: str,
    experiment_results: dict[str, dict],
    artifact_dir: Path,
) -> dict:
    """Export the full saved experiment package for one experiment."""
    selected_result = experiment_results[experiment_key]

    export_result = experiment_runner.export_best_model_to_artifacts(
        output_dir=selected_result["output_dir"],
        artifact_dir=artifact_dir,
    )

    print("Experiment key   :", experiment_key)
    print("Exported artifact:", export_result["destination_artifact_dir"])

    return export_result
