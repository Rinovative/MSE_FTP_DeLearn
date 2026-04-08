"""
Dataset and dataloader utilities for image classification.

This module provides helpers to:
- compute channel-wise mean/std from the training split,
- build train/validation transforms,
- create ImageFolder datasets,
- construct train/validation dataloaders, and
- package everything into a DataBundle that can be passed directly to
  ``run_experiment(...)``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Image tensor dimensions
IMAGE_NDIM = 4
LABEL_NDIM = 1


@dataclass(slots=True)
class DataBundle:
    """
    Container for all data objects required by the training pipeline.

    Attributes
    ----------
    train_loader
        Training dataloader.
    val_loader
        Validation dataloader.
    class_names
        Ordered class names.
    data_config
        Serializable metadata used for experiment logging.
    train_dataset
        Training dataset.
    val_dataset
        Validation dataset.
    mean
        Channel-wise training mean.
    std
        Channel-wise training standard deviation.

    """

    train_loader: DataLoader
    val_loader: DataLoader
    class_names: list[str]
    data_config: dict[str, Any]
    train_dataset: datasets.ImageFolder
    val_dataset: datasets.ImageFolder
    mean: list[float]
    std: list[float]

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the bundle to a dict-like structure if needed elsewhere.

        Returns
        -------
        dict[str, Any]
            Dictionary containing the main bundle fields.

        """
        return {
            "train_loader": self.train_loader,
            "val_loader": self.val_loader,
            "class_names": self.class_names,
            "data_config": self.data_config,
            "train_dataset": self.train_dataset,
            "val_dataset": self.val_dataset,
            "mean": self.mean,
            "std": self.std,
        }


def _resolve_split_dirs(data_root: str | Path) -> tuple[Path, Path]:
    """
    Resolve and validate the expected train/validation split directories.

    Expected structure
    ------------------
    data_root/
        train/
            class_a/
            class_b/
            ...
        validate/
            class_a/
            class_b/
            ...

    Parameters
    ----------
    data_root
        Root directory containing ``train/`` and ``validate/``.

    Returns
    -------
    tuple[Path, Path]
        Training directory and validation directory.

    """
    data_root = Path(data_root)

    train_dir = data_root / "train"
    val_dir = data_root / "validate"

    if not train_dir.exists():
        msg = f"Train directory not found: {train_dir}"
        raise FileNotFoundError(msg)

    if not val_dir.exists():
        msg = f"Validation directory not found: {val_dir}"
        raise FileNotFoundError(msg)

    return train_dir, val_dir


def _seed_worker(worker_id: int) -> None:
    """
    Seed python random inside each dataloader worker.

    Parameters
    ----------
    worker_id
        Worker id provided by PyTorch.

    Returns
    -------
    None
        Seeds the worker-local RNGs.

    """
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)


def _make_torch_generator(seed: int) -> torch.Generator:
    """
    Build a torch generator with a fixed manual seed.

    Parameters
    ----------
    seed
        Seed value for the generator.

    Returns
    -------
    torch.Generator
        Seeded generator.

    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def compute_train_mean_std(
    train_dir: str | Path,
    image_size: int = 128,
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> tuple[list[float], list[float]]:
    """
    Compute channel-wise mean and standard deviation over the training split.

    Only the training split is used. No augmentation is applied.
    Images are resized and converted to tensors before the statistics
    are accumulated over all pixels.

    Parameters
    ----------
    train_dir
        Path to the training folder containing class subfolders.
    image_size
        Target square image size used before conversion to tensor.
    batch_size
        Batch size for the temporary statistics loader.
    num_workers
        Number of worker processes.
    pin_memory
        Whether to enable ``pin_memory`` in the temporary loader.

    Returns
    -------
    tuple[list[float], list[float]]
        Channel-wise mean and standard deviation.

    """
    train_dir = Path(train_dir)

    if not train_dir.exists():
        msg = f"Train directory not found: {train_dir}"
        raise FileNotFoundError(msg)

    stats_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )

    stats_dataset = datasets.ImageFolder(train_dir, transform=stats_transform)

    stats_loader = DataLoader(
        stats_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

    channel_sum = torch.zeros(3, dtype=torch.float64)
    channel_sum_sq = torch.zeros(3, dtype=torch.float64)
    num_pixels = 0

    for batch_images_raw, _batch_labels in stats_loader:
        batch_images = batch_images_raw.to(dtype=torch.float64)
        batch_pixels = batch_images.size(0) * batch_images.size(2) * batch_images.size(3)

        channel_sum += batch_images.sum(dim=(0, 2, 3))
        channel_sum_sq += (batch_images**2).sum(dim=(0, 2, 3))
        num_pixels += batch_pixels

    mean = channel_sum / num_pixels
    std = torch.sqrt(channel_sum_sq / num_pixels - mean**2)

    return mean.tolist(), std.tolist()


def build_transforms(
    image_size: int,
    mean: list[float],
    std: list[float],
    use_flip: bool = True,
    randaugment_num_ops: int = 0,
    randaugment_magnitude: int = 9,
) -> tuple[transforms.Compose, transforms.Compose]:
    """
    Build training and validation transforms.

    Parameters
    ----------
    image_size
        Target square image size.
    mean
        Channel-wise training mean.
    std
        Channel-wise training standard deviation.
    use_flip
        Whether to apply ``RandomHorizontalFlip`` to the training set.
    randaugment_num_ops
        Number of RandAugment operations per image.
        If 0, RandAugment is disabled.
    randaugment_magnitude
        RandAugment magnitude.

    Returns
    -------
    tuple[transforms.Compose, transforms.Compose]
        Training transform and validation transform.

    """
    if mean is None or std is None:
        msg = "mean and std must be provided."
        raise ValueError(msg)

    normalize = transforms.Normalize(mean=mean, std=std)

    train_steps: list[Any] = [
        transforms.Resize((image_size, image_size)),
    ]

    if use_flip:
        train_steps.append(transforms.RandomHorizontalFlip(p=0.5))

    if randaugment_num_ops > 0:
        train_steps.append(
            transforms.RandAugment(
                num_ops=randaugment_num_ops,
                magnitude=randaugment_magnitude,
            )
        )

    train_steps.extend(
        [
            transforms.ToTensor(),
            normalize,
        ]
    )

    val_steps: list[Any] = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ]

    train_transform = transforms.Compose(train_steps)
    val_transform = transforms.Compose(val_steps)

    return train_transform, val_transform


def build_datasets(
    data_root: str | Path,
    image_size: int,
    mean: list[float],
    std: list[float],
    use_flip: bool = True,
    randaugment_num_ops: int = 0,
    randaugment_magnitude: int = 9,
) -> tuple[datasets.ImageFolder, datasets.ImageFolder, list[str]]:
    """
    Build training and validation datasets from the expected folder structure.

    Parameters
    ----------
    data_root
        Root directory containing ``train/`` and ``validate/``.
    image_size
        Target square image size.
    mean
        Channel-wise training mean.
    std
        Channel-wise training standard deviation.
    use_flip
        Whether to apply ``RandomHorizontalFlip`` to the training set.
    randaugment_num_ops
        Number of RandAugment operations per image.
    randaugment_magnitude
        RandAugment magnitude.

    Returns
    -------
    tuple[datasets.ImageFolder, datasets.ImageFolder, list[str]]
        Training dataset, validation dataset, and ordered class names.

    """
    train_dir, val_dir = _resolve_split_dirs(data_root)

    train_transform, val_transform = build_transforms(
        image_size=image_size,
        mean=mean,
        std=std,
        use_flip=use_flip,
        randaugment_num_ops=randaugment_num_ops,
        randaugment_magnitude=randaugment_magnitude,
    )

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    if train_dataset.classes != val_dataset.classes:
        msg = "Train and validation classes do not match."
        raise ValueError(msg)

    class_names = list(train_dataset.classes)

    return train_dataset, val_dataset, class_names


def build_dataloaders(
    data_root: str | Path,
    mean: list[float],
    std: list[float],
    image_size: int = 128,
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True,
    use_flip: bool = True,
    randaugment_num_ops: int = 0,
    randaugment_magnitude: int = 9,
    seed: int | None = None,
) -> tuple[DataLoader, DataLoader, list[str], datasets.ImageFolder, datasets.ImageFolder]:
    """
    Build training and validation dataloaders.

    The normalization statistics are not computed inside this function.
    They must be computed externally from the training split and passed in.

    Parameters
    ----------
    data_root
        Root directory containing ``train/`` and ``validate/``.
    mean
        Channel-wise training mean computed externally.
    std
        Channel-wise training standard deviation computed externally.
    image_size
        Target square image size.
    batch_size
        Batch size for training and validation loaders.
    num_workers
        Number of worker processes.
    pin_memory
        Whether to enable ``pin_memory`` in the loaders.
    use_flip
        Whether to apply ``RandomHorizontalFlip`` to the training set.
    randaugment_num_ops
        Number of RandAugment operations per image.
    randaugment_magnitude
        RandAugment magnitude.
    seed
        Optional seed for deterministic dataloader shuffling and worker RNGs.

    Returns
    -------
    tuple[DataLoader, DataLoader, list[str], datasets.ImageFolder, datasets.ImageFolder]
        Training loader, validation loader, class names,
        training dataset, and validation dataset.

    """
    train_dataset, val_dataset, class_names = build_datasets(
        data_root=data_root,
        image_size=image_size,
        mean=mean,
        std=std,
        use_flip=use_flip,
        randaugment_num_ops=randaugment_num_ops,
        randaugment_magnitude=randaugment_magnitude,
    )

    train_generator = None
    val_generator = None
    worker_init_fn = None

    if seed is not None:
        train_generator = _make_torch_generator(seed)
        val_generator = _make_torch_generator(seed + 1)
        worker_init_fn = _seed_worker

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        worker_init_fn=worker_init_fn,
        generator=train_generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        worker_init_fn=worker_init_fn,
        generator=val_generator,
    )

    return train_loader, val_loader, class_names, train_dataset, val_dataset


def build_data_bundle(
    data_root: str | Path,
    image_size: int = 128,
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True,
    use_flip: bool = True,
    randaugment_num_ops: int = 0,
    randaugment_magnitude: int = 9,
    mean: list[float] | None = None,
    std: list[float] | None = None,
    stats_batch_size: int | None = None,
    stats_num_workers: int | None = None,
    dataset_name: str | None = None,
    seed: int | None = None,
) -> DataBundle:
    """
    Build a full ``DataBundle`` for direct use with ``run_experiment(...)``.

    If ``mean`` and ``std`` are not provided, they are computed automatically
    from the training split.

    Parameters
    ----------
    data_root
        Root directory containing ``train/`` and ``validate/``.
    image_size
        Target square image size.
    batch_size
        Batch size for training and validation loaders.
    num_workers
        Number of worker processes for the training and validation loaders.
    pin_memory
        Whether to enable ``pin_memory``.
    use_flip
        Whether to apply ``RandomHorizontalFlip`` to the training set.
    randaugment_num_ops
        Number of RandAugment operations per image.
    randaugment_magnitude
        RandAugment magnitude.
    mean
        Optional precomputed channel-wise mean.
    std
        Optional precomputed channel-wise standard deviation.
    stats_batch_size
        Optional batch size used only for computing mean/std.
        If ``None``, ``batch_size`` is used.
    stats_num_workers
        Optional number of workers used only for computing mean/std.
        If ``None``, ``num_workers`` is used.
    dataset_name
        Optional custom dataset name for logging.
    seed
        Optional seed for deterministic dataloader shuffling and worker RNGs.

    Returns
    -------
    DataBundle
        Fully prepared data bundle compatible with ``run_experiment(...)``.

    """
    data_root = Path(data_root)
    train_dir, val_dir = _resolve_split_dirs(data_root)

    stats_batch_size = batch_size if stats_batch_size is None else stats_batch_size
    stats_num_workers = num_workers if stats_num_workers is None else stats_num_workers

    if mean is None or std is None:
        mean, std = compute_train_mean_std(
            train_dir=train_dir,
            image_size=image_size,
            batch_size=stats_batch_size,
            num_workers=stats_num_workers,
            pin_memory=pin_memory,
        )

    train_loader, val_loader, class_names, train_dataset, val_dataset = build_dataloaders(
        data_root=data_root,
        mean=mean,
        std=std,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        use_flip=use_flip,
        randaugment_num_ops=randaugment_num_ops,
        randaugment_magnitude=randaugment_magnitude,
        seed=seed,
    )

    resolved_dataset_name = dataset_name if dataset_name is not None else data_root.name

    data_config: dict[str, Any] = {
        "dataset_name": resolved_dataset_name,
        "data_root": str(data_root),
        "train_dir": str(train_dir),
        "val_dir": str(val_dir),
        "image_size": image_size,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "use_flip": use_flip,
        "randaugment_num_ops": randaugment_num_ops,
        "randaugment_magnitude": randaugment_magnitude,
        "mean": mean,
        "std": std,
        "stats_batch_size": stats_batch_size,
        "stats_num_workers": stats_num_workers,
        "class_to_idx": dict(train_dataset.class_to_idx),
        "seed": seed,
    }

    return DataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        class_names=class_names,
        data_config=data_config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        mean=mean,
        std=std,
    )


def denormalize_image(
    image: torch.Tensor,
    mean: list[float],
    std: list[float],
) -> torch.Tensor:
    """
    Undo normalization for visualization.

    Parameters
    ----------
    image
        Image tensor of shape ``[C, H, W]``.
    mean
        Channel-wise mean.
    std
        Channel-wise standard deviation.

    Returns
    -------
    torch.Tensor
        Denormalized image tensor of shape ``[C, H, W]``.

    """
    mean_tensor = torch.tensor(mean, dtype=image.dtype, device=image.device).view(-1, 1, 1)
    std_tensor = torch.tensor(std, dtype=image.dtype, device=image.device).view(-1, 1, 1)

    image_denorm = image * std_tensor + mean_tensor
    return image_denorm.clamp(0.0, 1.0)


def show_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    class_names: list[str],
    mean: list[float] | None = None,
    std: list[float] | None = None,
    max_images: int = 8,
    figsize: tuple[int, int] = (16, 8),
) -> None:
    """
    Show a small batch of images with labels.

    Parameters
    ----------
    images
        Batch tensor of shape ``[B, C, H, W]``.
    labels
        Label tensor of shape ``[B]``.
    class_names
        Class names indexed by label.
    mean
        Optional normalization mean for denormalization.
    std
        Optional normalization standard deviation for denormalization.
    max_images
        Maximum number of images to show.
    figsize
        Figure size.

    Returns
    -------
    None
        Displays the figure.

    """
    if images.ndim != IMAGE_NDIM:
        msg = "images must have shape [B, C, H, W]."
        raise ValueError(msg)

    if labels.ndim != LABEL_NDIM:
        msg = "labels must have shape [B]."
        raise ValueError(msg)

    num_images = min(max_images, images.size(0))

    if num_images <= 0:
        msg = "At least one image is required for visualization."
        raise ValueError(msg)

    _fig, axes = plt.subplots(1, num_images, figsize=figsize)

    if num_images == 1:
        axes = [axes]

    for idx in range(num_images):
        image_cpu = images[idx].detach().cpu()

        if mean is not None and std is not None:
            image_cpu = denormalize_image(image_cpu, mean=mean, std=std)

        image_np = image_cpu.permute(1, 2, 0).numpy()
        label_idx = int(labels[idx].item())

        axes[idx].imshow(image_np)
        axes[idx].set_title(class_names[label_idx])
        axes[idx].axis("off")

    plt.tight_layout()
    plt.show()
