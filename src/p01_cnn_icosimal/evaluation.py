"""
Evaluation utilities for CNN experiments on the iCoSimal dataset.

This module provides:
- loading saved experiment outputs,
- collecting predictions from a dataloader,
- reusable plotting functions,
- a notebook widget with one central control area,
- export of the currently visible plot.

Design goals:
- every plot is its own function,
- one widget bundles all views,
- controls are shown dynamically depending on the selected view,
- current plot can be exported directly from the widget.
"""

from __future__ import annotations

import json
import math
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import FileLink, clear_output, display
from ipywidgets import HTML, Button, Checkbox, Dropdown, HBox, Image, Label, Layout, Output, ToggleButton, VBox
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.nn import functional

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def _to_device(device: torch.device | str) -> torch.device:
    """Convert a device specification to ``torch.device``."""
    if isinstance(device, torch.device):
        return device
    return torch.device(device)


def _load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON file from disk."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _ensure_output_dir(output_dir: str | Path) -> Path:
    """Validate and normalize an experiment output directory."""
    output_dir = Path(output_dir)

    if not output_dir.exists():
        msg = f"Output directory does not exist: {output_dir}"
        raise FileNotFoundError(msg)

    if not output_dir.is_dir():
        msg = f"Output path is not a directory: {output_dir}"
        raise NotADirectoryError(msg)

    return output_dir


def _get_result_output_dir(result: dict[str, Any]) -> Path:
    """Extract and validate ``output_dir`` from a result dictionary."""
    if "output_dir" not in result:
        msg = "Result dictionary does not contain 'output_dir'."
        raise KeyError(msg)

    return _ensure_output_dir(result["output_dir"])


def load_saved_experiment(output_dir: str | Path) -> dict[str, Any]:
    """
    Load a saved experiment from disk.

    Expected files:
    - config.json
    - summary.json
    - history.json
    """
    output_dir = _ensure_output_dir(output_dir)

    config_path = output_dir / "config.json"
    summary_path = output_dir / "summary.json"
    history_path = output_dir / "history.json"

    if not config_path.exists():
        msg = f"Missing config.json in {output_dir}"
        raise FileNotFoundError(msg)

    if not summary_path.exists():
        msg = f"Missing summary.json in {output_dir}"
        raise FileNotFoundError(msg)

    if not history_path.exists():
        msg = f"Missing history.json in {output_dir}"
        raise FileNotFoundError(msg)

    return {
        "output_dir": str(output_dir),
        "config": _load_json(config_path),
        "summary": _load_json(summary_path),
        "history": _load_json(history_path),
    }


def load_saved_experiment_from_result(result: dict[str, Any]) -> dict[str, Any]:
    """Load a saved experiment using ``result['output_dir']``."""
    return load_saved_experiment(_get_result_output_dir(result))


def _extract_history(result_or_saved: dict[str, Any]) -> dict[str, list[float]]:
    """Extract history from either an in-memory result or a loaded saved experiment."""
    if "history" in result_or_saved:
        return result_or_saved["history"]

    msg = "Could not find 'history' in the provided object."
    raise KeyError(msg)


def _extract_class_names(data_bundle: Any = None, saved_experiment: dict[str, Any] | None = None) -> list[str]:
    """Extract class names from either the data bundle or the saved config."""
    if data_bundle is not None and hasattr(data_bundle, "class_names"):
        return list(data_bundle.class_names)

    if data_bundle is not None and isinstance(data_bundle, dict) and "class_names" in data_bundle:
        return list(data_bundle["class_names"])

    if saved_experiment is not None:
        config = saved_experiment.get("config", {})
        class_names = config.get("data", {}).get("dataset", {}).get("class_names")
        if class_names is not None:
            return list(class_names)

    msg = "Could not determine class names."
    raise ValueError(msg)


def _extract_mean_std(data_bundle: Any = None, saved_experiment: dict[str, Any] | None = None) -> tuple[list[float], list[float]]:
    """Extract normalization mean and std from the data bundle or saved config."""
    if data_bundle is not None:
        mean = getattr(data_bundle, "mean", None) if not isinstance(data_bundle, dict) else data_bundle.get("mean")
        std = getattr(data_bundle, "std", None) if not isinstance(data_bundle, dict) else data_bundle.get("std")
        if mean is not None and std is not None:
            return list(mean), list(std)

    if saved_experiment is not None:
        norm = saved_experiment.get("config", {}).get("data", {}).get("normalization", {})
        mean = norm.get("mean")
        std = norm.get("std")
        if mean is not None and std is not None:
            return list(mean), list(std)

    msg = "Could not determine normalization mean/std."
    raise ValueError(msg)


def _denormalize_images(
    images: torch.Tensor,
    mean: list[float],
    std: list[float],
) -> torch.Tensor:
    """Denormalize a batch of images for visualization."""
    mean_tensor = torch.tensor(mean, device=images.device).view(1, -1, 1, 1)
    std_tensor = torch.tensor(std, device=images.device).view(1, -1, 1, 1)
    return images * std_tensor + mean_tensor


def _sanitize_name_for_path(name: str) -> str:
    """Make a name safe for filesystem paths."""
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in name)
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe.strip("_")


def _model_display_name(model: torch.nn.Module) -> str:
    """Return a compact model display name for the widget title."""
    return model.__class__.__name__


def _apply_button_style(button: Button | ToggleButton, *, role: str) -> None:
    """Apply consistent button styling."""
    role_to_colors = {
        "primary": {"button_style": "info", "button_color": "#1f77b4"},
        "success": {"button_style": "success", "button_color": "#2ca02c"},
        "danger": {"button_style": "danger", "button_color": "#d62728"},
    }

    selected = role_to_colors[role]

    button.button_style = selected["button_style"]
    if hasattr(button.style, "button_color"):
        button.style.button_color = selected["button_color"]

    button.style.font_weight = "bold"
    button.style.text_color = "white"
    button.layout.height = "38px"
    button.layout.border = "1px solid #c9d1d9"


def _figure_to_widget(fig: Figure, *, max_width: str = "950px") -> Image:
    """Render a Matplotlib figure into a compact ipywidgets Image widget."""
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=130, bbox_inches="tight")
    png_bytes = buffer.getvalue()
    buffer.close()

    return Image(
        value=png_bytes,
        format="png",
        layout=Layout(
            width="100%",
            max_width=max_width,
            height="auto",
            overflow="hidden",
            object_fit="contain",
            align_self="center",
        ),
    )


@torch.inference_mode()
def collect_predictions(
    *,
    model: torch.nn.Module,
    dataloader: Any,
    device: torch.device | str,
    return_images: bool = False,
    max_samples: int | None = None,
) -> dict[str, Any]:
    """Collect predictions, probabilities, and optionally images from a dataloader."""
    device = _to_device(device)
    model = model.to(device)
    model.eval()

    all_targets: list[torch.Tensor] = []
    all_predictions: list[torch.Tensor] = []
    all_probabilities: list[torch.Tensor] = []
    all_confidences: list[torch.Tensor] = []
    all_logits: list[torch.Tensor] = []
    all_images: list[torch.Tensor] = []

    collected = 0

    for batch_images, batch_targets in dataloader:
        images = batch_images.to(device, non_blocking=True)
        targets = batch_targets.to(device, non_blocking=True)

        logits = model(images)
        probabilities = functional.softmax(logits, dim=1)
        confidences, predictions = probabilities.max(dim=1)

        all_targets.append(targets.detach().cpu())
        all_predictions.append(predictions.detach().cpu())
        all_probabilities.append(probabilities.detach().cpu())
        all_confidences.append(confidences.detach().cpu())
        all_logits.append(logits.detach().cpu())

        if return_images:
            all_images.append(batch_images.detach().cpu())

        collected += targets.size(0)

        if max_samples is not None and collected >= max_samples:
            break

    targets_tensor = torch.cat(all_targets, dim=0)
    predictions_tensor = torch.cat(all_predictions, dim=0)
    probabilities_tensor = torch.cat(all_probabilities, dim=0)
    confidences_tensor = torch.cat(all_confidences, dim=0)
    logits_tensor = torch.cat(all_logits, dim=0)

    if max_samples is not None:
        targets_tensor = targets_tensor[:max_samples]
        predictions_tensor = predictions_tensor[:max_samples]
        probabilities_tensor = probabilities_tensor[:max_samples]
        confidences_tensor = confidences_tensor[:max_samples]
        logits_tensor = logits_tensor[:max_samples]

    result: dict[str, Any] = {
        "targets": targets_tensor.numpy(),
        "predictions": predictions_tensor.numpy(),
        "probabilities": probabilities_tensor.numpy(),
        "confidences": confidences_tensor.numpy(),
        "logits": logits_tensor.numpy(),
    }

    if return_images:
        images_tensor = torch.cat(all_images, dim=0)
        if max_samples is not None:
            images_tensor = images_tensor[:max_samples]
        result["images"] = images_tensor

    return result


def evaluate_model_on_dataloader(
    *,
    model: torch.nn.Module,
    dataloader: Any,
    device: torch.device | str,
    class_names: list[str],
    return_images: bool = True,
    max_samples: int | None = None,
) -> dict[str, Any]:
    """Run a full evaluation pass and return reusable evaluation data."""
    pred_data = collect_predictions(
        model=model,
        dataloader=dataloader,
        device=device,
        return_images=return_images,
        max_samples=max_samples,
    )

    targets = pred_data["targets"]
    predictions = pred_data["predictions"]

    pred_data["correct_mask"] = predictions == targets
    pred_data["class_names"] = class_names
    return pred_data


def compute_classification_report_dict(
    *,
    targets: np.ndarray,
    predictions: np.ndarray,
    class_names: list[str],
) -> dict[str, Any]:
    """Compute per-class precision, recall, f1, and support."""
    precision, recall, f1, support = precision_recall_fscore_support(
        targets,
        predictions,
        labels=np.arange(len(class_names)),
        zero_division=0,
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
        "class_names": np.array(class_names),
    }


def create_learning_curves_figure(
    history: dict[str, list[float]],
) -> tuple[Figure, Axes]:
    """Plot train/validation loss and accuracy curves in one figure."""
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    ax.plot(epochs, history["train_loss"], label="Train loss")
    ax.plot(epochs, history["val_loss"], label="Val loss")
    ax.plot(epochs, history["train_accuracy"], label="Train accuracy")
    ax.plot(epochs, history["val_accuracy"], label="Val accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Metric value")
    ax.set_title("Learning curves")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def create_prediction_confidence_histogram_figure(
    *,
    confidences: np.ndarray,
    correct_mask: np.ndarray,
    bins: int = 20,
) -> tuple[Figure, Axes]:
    """Create a confidence histogram for correct vs. wrong predictions."""
    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    ax.hist(confidences[correct_mask], bins=bins, alpha=0.7, label="Correct")
    ax.hist(confidences[~correct_mask], bins=bins, alpha=0.7, label="Wrong")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Count")
    ax.set_title("Prediction confidence histogram")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def create_confusion_matrix_figure(
    *,
    targets: np.ndarray,
    predictions: np.ndarray,
    class_names: list[str],
    normalize: bool = False,
) -> tuple[Figure, Axes]:
    """Create a confusion matrix figure."""
    cm = confusion_matrix(targets, predictions, labels=np.arange(len(class_names)))

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm = cm / row_sums

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(cm, aspect="auto")
    fig.colorbar(im, ax=ax)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    fmt = ".2f" if normalize else "d"
    threshold = cm.max() / 2 if cm.size > 0 else 0.0

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = format(cm[i, j], fmt)
            ax.text(
                j,
                i,
                value,
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
                fontsize=8,
            )

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion matrix" + (" (normalized)" if normalize else ""))
    fig.tight_layout()
    return fig, ax


def create_top_misclassifications_figure(
    *,
    targets: np.ndarray,
    predictions: np.ndarray,
    class_names: list[str],
    top_k: int = 15,
) -> tuple[Figure, Axes]:
    """Create a horizontal bar chart of the most frequent misclassification pairs."""
    num_classes = len(class_names)
    cm = confusion_matrix(targets, predictions, labels=np.arange(num_classes))

    errors: list[tuple[str, str, int]] = []
    for true_idx in range(num_classes):
        for pred_idx in range(num_classes):
            if true_idx == pred_idx:
                continue

            count = int(cm[true_idx, pred_idx])
            if count > 0:
                errors.append((class_names[true_idx], class_names[pred_idx], count))

    errors = sorted(errors, key=lambda item: item[2], reverse=True)[:top_k]

    fig, ax = plt.subplots(figsize=(9.2, 4.6))

    if not errors:
        ax.axis("off")
        ax.text(0.5, 0.5, "No misclassifications found.", ha="center", va="center", fontsize=12)
        fig.tight_layout()
        return fig, ax

    labels = [f"{true_name} → {pred_name}" for true_name, pred_name, _ in errors]
    counts = [count for _, _, count in errors]

    ax.barh(labels[::-1], counts[::-1])
    ax.set_xlabel("Count")
    ax.set_title(f"Top {len(errors)} misclassification pairs")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    return fig, ax


def create_class_metrics_figure(
    *,
    targets: np.ndarray,
    predictions: np.ndarray,
    class_names: list[str],
) -> tuple[Figure, Axes]:
    """Create a per-class precision/recall/F1 figure."""
    report = compute_classification_report_dict(
        targets=targets,
        predictions=predictions,
        class_names=class_names,
    )

    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    ax.bar(x - width, report["precision"], width=width, label="Precision")
    ax.bar(x, report["recall"], width=width, label="Recall")
    ax.bar(x + width, report["f1"], width=width, label="F1")

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Per-class metrics")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def _filter_example_indices(
    *,
    targets: np.ndarray,
    predictions: np.ndarray,
    selected_class_idx: int | None,
    only_wrong: bool,
) -> np.ndarray:
    """Filter indices for the examples view using the true class only."""
    indices = np.arange(len(targets))

    if selected_class_idx is not None:
        true_class_mask = targets == selected_class_idx
        indices = indices[true_class_mask]

    if only_wrong:
        wrong_mask = targets[indices] != predictions[indices]
        indices = indices[wrong_mask]

    return indices


def create_examples_figure(
    *,
    images: torch.Tensor,
    targets: np.ndarray,
    predictions: np.ndarray,
    confidences: np.ndarray,
    class_names: list[str],
    mean: list[float],
    std: list[float],
    selected_class_idx: int | None = None,
    only_wrong: bool = False,
    page: int = 0,
    images_per_page: int = 8,
    cols: int = 4,
) -> tuple[Figure, np.ndarray, dict[str, Any]]:
    """
    Create an examples figure with paging support.

    Returns
    -------
    tuple
        (fig, axes_array, metadata) where metadata includes page counters.

    """
    filtered_indices = _filter_example_indices(
        targets=targets,
        predictions=predictions,
        selected_class_idx=selected_class_idx,
        only_wrong=only_wrong,
    )

    total_matching = len(filtered_indices)
    total_pages = max(1, math.ceil(total_matching / images_per_page))

    page = max(0, min(page, total_pages - 1))
    start = page * images_per_page
    end = start + images_per_page
    page_indices = filtered_indices[start:end]

    rows = max(1, math.ceil(max(len(page_indices), 1) / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(9.4, 2.8 * rows))
    axes_array = np.atleast_1d(axes).ravel()

    denorm_images = _denormalize_images(images, mean=mean, std=std).clamp(0.0, 1.0)

    title_parts = ["Prediction examples"]
    if selected_class_idx is not None:
        title_parts.append(f"class={class_names[selected_class_idx]}")
    if only_wrong:
        title_parts.append("wrong only")
    title_parts.append(f"page {page + 1}/{total_pages}")

    fig.suptitle(" | ".join(title_parts), fontsize=13, y=0.98)

    if total_matching == 0:
        for ax in axes_array:
            ax.axis("off")
        axes_array[0].text(0.5, 0.5, "No matching samples found.", ha="center", va="center", fontsize=12)
        fig.subplots_adjust(top=0.82, bottom=0.08, left=0.04, right=0.98, hspace=0.30, wspace=0.15)
        metadata = {
            "page": page,
            "total_pages": total_pages,
            "total_matching": total_matching,
            "page_indices": page_indices,
        }
        return fig, axes_array, metadata

    for ax_idx, ax in enumerate(axes_array):
        ax.axis("off")

        if ax_idx >= len(page_indices):
            continue

        sample_idx = page_indices[ax_idx]
        image = denorm_images[sample_idx].permute(1, 2, 0).cpu().numpy()

        true_idx = int(targets[sample_idx])
        pred_idx = int(predictions[sample_idx])
        conf = float(confidences[sample_idx])

        ax.imshow(image)
        ax.set_title(
            f"T: {class_names[true_idx]}\nP: {class_names[pred_idx]} ({conf:.2f})",
            fontsize=8.5,
            pad=6,
        )

    fig.subplots_adjust(top=0.88, bottom=0.06, left=0.03, right=0.98, hspace=0.18, wspace=0.10)

    metadata = {
        "page": page,
        "total_pages": total_pages,
        "total_matching": total_matching,
        "page_indices": page_indices,
    }
    return fig, axes_array, metadata


def _save_current_figure(
    *,
    fig: Figure,
    export_dir: str | Path,
    filename_stem: str,
) -> Path:
    """Save the current figure to disk."""
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    safe_name = _sanitize_name_for_path(filename_stem)
    destination = export_dir / f"{safe_name}.png"

    counter = 2
    while destination.exists():
        destination = export_dir / f"{safe_name}_{counter}.png"
        counter += 1

    fig.savefig(destination, dpi=200, bbox_inches="tight")
    return destination


def create_evaluation_widget(  # noqa: C901, PLR0915
    *,
    result_or_saved: dict[str, Any],
    model: torch.nn.Module,
    dataloader: Any,
    device: torch.device | str,
    data_bundle: Any | None = None,
    max_samples: int | None = None,
    export_dir: str | Path | None = None,
) -> VBox:
    """
    Create the evaluation widget.

    Behaviour:
    - first controlled by one main toggle button,
    - inside it a dropdown selects the active view,
    - controls adapt to the active view,
    - current plot can be exported.
    """
    history = _extract_history(result_or_saved)

    saved_experiment = None
    if "config" in result_or_saved and "summary" in result_or_saved:
        saved_experiment = result_or_saved
    elif "output_dir" in result_or_saved:
        try:
            saved_experiment = load_saved_experiment(result_or_saved["output_dir"])
        except (FileNotFoundError, NotADirectoryError, KeyError):
            saved_experiment = None

    class_names = _extract_class_names(data_bundle=data_bundle, saved_experiment=saved_experiment)
    mean, std = _extract_mean_std(data_bundle=data_bundle, saved_experiment=saved_experiment)

    eval_data = evaluate_model_on_dataloader(
        model=model,
        dataloader=dataloader,
        device=device,
        class_names=class_names,
        return_images=True,
        max_samples=max_samples,
    )

    if export_dir is None:
        if "output_dir" in result_or_saved and result_or_saved["output_dir"] is not None:
            export_dir = Path(result_or_saved["output_dir"]) / "evaluation_exports"
        else:
            export_dir = Path.cwd() / "evaluation_exports"

    current_figure: dict[str, Any] = {"fig": None, "filename_stem": None}
    examples_state: dict[str, int] = {"page": 0}
    images_per_page = 8
    model_name = _model_display_name(model)

    root_output = Output(layout=Layout(width="100%"))
    controls_output = Output(layout=Layout(width="100%"))
    plot_output = Output(
        layout=Layout(
            width="100%",
            max_width="980px",
            overflow="hidden",
            align_self="center",
        )
    )
    export_output = Output(layout=Layout(width="100%"))

    main_container = VBox(
        [],
        layout=Layout(width="100%", align_items="stretch"),
    )

    open_button = ToggleButton(
        value=False,
        description=f"Open {model_name} evaluation",
        layout=Layout(width="260px"),
    )
    _apply_button_style(open_button, role="primary")

    view_dropdown = Dropdown(
        options=[
            ("Learning curves", "learning_curves"),
            ("Confidence histogram", "confidence_histogram"),
            ("Confusion matrix", "confusion_matrix"),
            ("Top misclassifications", "top_misclassifications"),
            ("Class metrics", "class_metrics"),
            ("Examples", "examples"),
        ],
        value="learning_curves",
        description="View:",
        layout=Layout(width="340px"),
    )

    normalize_checkbox = Checkbox(
        value=False,
        description="Normalize",
        indent=False,
    )

    class_dropdown = Dropdown(
        options=[("All classes", None)] + [(class_name, idx) for idx, class_name in enumerate(class_names)],
        value=None,
        description="Class:",
        layout=Layout(width="320px"),
    )

    wrong_only_checkbox = Checkbox(
        value=False,
        description="Wrong only",
        indent=False,
    )

    prev_button = Button(
        description="← Prev",
        layout=Layout(width="100px"),
    )
    _apply_button_style(prev_button, role="danger")

    next_button = Button(
        description="Next →",
        layout=Layout(width="100px"),
    )
    _apply_button_style(next_button, role="success")

    page_label = Label(value="Page 1/1")

    export_button = Button(
        description="Export current plot",
        layout=Layout(width="180px"),
    )
    _apply_button_style(export_button, role="success")

    spacer = HTML("<div style='height: 6px;'></div>")

    def _close_previous_figure() -> None:
        fig = current_figure.get("fig")
        if fig is not None:
            plt.close(fig)
            current_figure["fig"] = None
            current_figure["filename_stem"] = None

    def _set_current_figure(fig: Figure, filename_stem: str) -> None:
        _close_previous_figure()
        current_figure["fig"] = fig
        current_figure["filename_stem"] = filename_stem

    def _show_plot_widget(fig: Figure) -> None:
        plot_widget = _figure_to_widget(fig)

        with plot_output:
            clear_output(wait=True)
            display(plot_widget)

        plt.close(fig)

    def _render_controls() -> None:
        with controls_output:
            clear_output(wait=True)

            current_view = view_dropdown.value
            common_row = HBox([view_dropdown, export_button])

            if current_view == "confusion_matrix":
                display(VBox([common_row, HBox([normalize_checkbox])]))
                return

            if current_view == "examples":
                display(
                    VBox(
                        [
                            common_row,
                            HBox([class_dropdown, wrong_only_checkbox]),
                            HBox([prev_button, next_button, page_label]),
                        ]
                    )
                )
                return

            display(common_row)

    def _render_plot() -> None:
        current_view = view_dropdown.value

        with plt.ioff():
            if current_view == "learning_curves":
                fig, _ = create_learning_curves_figure(history)
                _set_current_figure(fig, "learning_curves")
                _show_plot_widget(fig)
                return

            if current_view == "confidence_histogram":
                fig, _ = create_prediction_confidence_histogram_figure(
                    confidences=eval_data["confidences"],
                    correct_mask=eval_data["correct_mask"],
                )
                _set_current_figure(fig, "confidence_histogram")
                _show_plot_widget(fig)
                return

            if current_view == "confusion_matrix":
                fig, _ = create_confusion_matrix_figure(
                    targets=eval_data["targets"],
                    predictions=eval_data["predictions"],
                    class_names=class_names,
                    normalize=normalize_checkbox.value,
                )
                suffix = "normalized" if normalize_checkbox.value else "raw"
                _set_current_figure(fig, f"confusion_matrix_{suffix}")
                _show_plot_widget(fig)
                return

            if current_view == "top_misclassifications":
                fig, _ = create_top_misclassifications_figure(
                    targets=eval_data["targets"],
                    predictions=eval_data["predictions"],
                    class_names=class_names,
                )
                _set_current_figure(fig, "top_misclassifications")
                _show_plot_widget(fig)
                return

            if current_view == "class_metrics":
                fig, _ = create_class_metrics_figure(
                    targets=eval_data["targets"],
                    predictions=eval_data["predictions"],
                    class_names=class_names,
                )
                _set_current_figure(fig, "class_metrics")
                _show_plot_widget(fig)
                return

            if current_view == "examples":
                fig, _, metadata = create_examples_figure(
                    images=eval_data["images"],
                    targets=eval_data["targets"],
                    predictions=eval_data["predictions"],
                    confidences=eval_data["confidences"],
                    class_names=class_names,
                    mean=mean,
                    std=std,
                    selected_class_idx=class_dropdown.value,
                    only_wrong=wrong_only_checkbox.value,
                    page=examples_state["page"],
                    images_per_page=images_per_page,
                )

                examples_state["page"] = metadata["page"]
                page_label.value = f"Page {metadata['page'] + 1}/{metadata['total_pages']}"
                prev_button.disabled = metadata["page"] <= 0
                next_button.disabled = metadata["page"] >= metadata["total_pages"] - 1

                selected_class_name = "all_classes" if class_dropdown.value is None else class_names[class_dropdown.value]
                wrong_suffix = "wrong_only" if wrong_only_checkbox.value else "all"
                _set_current_figure(fig, f"examples_{selected_class_name}_{wrong_suffix}_page_{metadata['page'] + 1}")
                _show_plot_widget(fig)
                return

        msg = f"Unknown view: {current_view}"
        raise ValueError(msg)

    def _refresh() -> None:
        _render_controls()
        _render_plot()

        main_container.children = (
            controls_output,
            spacer,
            plot_output,
            export_output,
        )

    def _hide_contents() -> None:
        _close_previous_figure()

        with controls_output:
            clear_output(wait=True)
        with plot_output:
            clear_output(wait=True)
        with export_output:
            clear_output(wait=True)
        with root_output:
            clear_output(wait=True)

        main_container.children = ()

    def _on_open_toggle(change: dict[str, Any]) -> None:
        if change["name"] != "value":
            return

        if open_button.value:
            open_button.description = f"Hide {model_name} evaluation"
            with root_output:
                clear_output(wait=True)
                display(main_container)
            _refresh()
        else:
            open_button.description = f"Open {model_name} evaluation"
            _hide_contents()

    def _on_view_change(change: dict[str, Any]) -> None:
        if change["name"] != "value":
            return

        if view_dropdown.value == "examples":
            examples_state["page"] = 0

        if open_button.value:
            _refresh()

    def _on_normalize_change(change: dict[str, Any]) -> None:
        if change["name"] != "value":
            return

        if open_button.value and view_dropdown.value == "confusion_matrix":
            _render_plot()

    def _on_examples_filter_change(change: dict[str, Any]) -> None:
        if change["name"] != "value":
            return

        if open_button.value and view_dropdown.value == "examples":
            examples_state["page"] = 0
            _render_plot()

    def _on_prev_click(_: Button) -> None:
        if not open_button.value or view_dropdown.value != "examples":
            return

        examples_state["page"] = max(0, examples_state["page"] - 1)
        _render_plot()

    def _on_next_click(_: Button) -> None:
        if not open_button.value or view_dropdown.value != "examples":
            return

        examples_state["page"] = examples_state["page"] + 1
        _render_plot()

    def _on_export_click(_: Button) -> None:
        with export_output:
            clear_output(wait=True)

            fig = current_figure.get("fig")
            filename_stem = current_figure.get("filename_stem")

            if fig is None or filename_stem is None:
                print("No current plot available for export.")
                return

            export_path = _save_current_figure(
                fig=fig,
                export_dir=export_dir,
                filename_stem=filename_stem,
            )

            print(f"Exported: {export_path}")
            display(FileLink(str(export_path)))

    open_button.observe(_on_open_toggle, names="value")
    view_dropdown.observe(_on_view_change, names="value")
    normalize_checkbox.observe(_on_normalize_change, names="value")
    class_dropdown.observe(_on_examples_filter_change, names="value")
    wrong_only_checkbox.observe(_on_examples_filter_change, names="value")

    prev_button.on_click(_on_prev_click)
    next_button.on_click(_on_next_click)
    export_button.on_click(_on_export_click)

    return VBox([open_button, root_output], layout=Layout(width="100%"))
