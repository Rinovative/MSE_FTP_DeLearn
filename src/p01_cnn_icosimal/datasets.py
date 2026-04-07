"""
Dataset and dataloader utilities for image classification.

This module provides helpers to:
- compute channel-wise mean/std from the training split,
- build train/validation transforms,
- create ImageFolder datasets, and
- construct train/validation dataloaders.
"""

from __future__ import annotations

from pathlib import Path

import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def compute_train_mean_std(
    train_dir: str | Path,
    image_size: int = 128,
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> tuple[list[float], list[float]]:
    """
    Compute channel-wise mean and std over the entire training split.

    Only the training split is used. No augmentation is applied.
    Images are resized and converted to tensors before the statistics
    are accumulated over all pixels.

    Parameters
    ----------
    train_dir : str | Path
        Path to the training folder containing class subfolders.
    image_size : int, default=128
        Target square image size used before conversion to tensor.
    batch_size : int, default=64
        Batch size for the temporary statistics loader.
    num_workers : int, default=4
        Number of worker processes.
    pin_memory : bool, default=True
        Whether to enable pin_memory in the temporary loader.

    Returns
    -------
    tuple[list[float], list[float]]
        Channel-wise mean and std.

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
    Build train and validation transforms.

    Parameters
    ----------
    image_size : int
        Target square image size.
    mean : list[float]
        Channel-wise training mean.
    std : list[float]
        Channel-wise training std.
    use_flip : bool, default=True
        Whether to apply RandomHorizontalFlip to the training set.
    randaugment_num_ops : int, default=0
        Number of RandAugment operations per image.
        If 0, RandAugment is disabled.
    randaugment_magnitude : int, default=9
        RandAugment magnitude.

    Returns
    -------
    tuple[transforms.Compose, transforms.Compose]
        Train and validation transforms.

    """
    if mean is None or std is None:
        msg = "mean and std must be provided"
        raise ValueError(msg)

    normalize = transforms.Normalize(mean=mean, std=std)

    train_steps: list[object] = [
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

    val_steps: list[object] = [
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
    Build train and validation datasets from the expected folder structure.

    Expected structure
    ------------------
    data_root/
        train/
            class_name_1/
            class_name_2/
            ...
        validate/
            class_name_1/
            class_name_2/
            ...

    Parameters
    ----------
    data_root : str | Path
        Root directory containing train/ and validate/.
    image_size : int
        Target square image size.
    mean : list[float]
        Channel-wise training mean.
    std : list[float]
        Channel-wise training std.
    use_flip : bool, default=True
        Whether to apply RandomHorizontalFlip to the training set.
    randaugment_num_ops : int, default=0
        Number of RandAugment operations per image.
    randaugment_magnitude : int, default=9
        RandAugment magnitude.

    Returns
    -------
    tuple[datasets.ImageFolder, datasets.ImageFolder, list[str]]
        Train dataset, validation dataset, and class names.

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

    class_names = train_dataset.classes

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
) -> tuple[DataLoader, DataLoader, list[str]]:
    """
    Build train and validation dataloaders.

    The normalization statistics are not computed inside this function.
    They must be computed externally from the training split and passed in.

    Parameters
    ----------
    data_root : str | Path
        Root directory containing train/ and validate/.
    mean : list[float]
        Channel-wise training mean computed externally.
    std : list[float]
        Channel-wise training std computed externally.
    image_size : int, default=128
        Target square image size.
    batch_size : int, default=64
        Batch size for train and validation loaders.
    num_workers : int, default=4
        Number of worker processes.
    pin_memory : bool, default=True
        Whether to enable pin_memory in the loaders.
    use_flip : bool, default=True
        Whether to apply RandomHorizontalFlip to the training set.
    randaugment_num_ops : int, default=0
        Number of RandAugment operations per image.
    randaugment_magnitude : int, default=9
        RandAugment magnitude.

    Returns
    -------
    tuple[DataLoader, DataLoader, list[str]]
        Train loader, validation loader, and class names.

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

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

    return train_loader, val_loader, class_names


def denormalize_image(
    image: torch.Tensor,
    mean: list[float],
    std: list[float],
) -> torch.Tensor:
    """
    Undo normalization for visualization.

    Parameters
    ----------
    image : torch.Tensor
        Image tensor of shape [C, H, W].
    mean : list[float]
        Channel-wise mean.
    std : list[float]
        Channel-wise std.

    Returns
    -------
    torch.Tensor
        Denormalized image tensor of shape [C, H, W].

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
    images : torch.Tensor
        Batch tensor of shape [B, C, H, W].
    labels : torch.Tensor
        Label tensor of shape [B].
    class_names : list[str]
        Class names indexed by label.
    mean : list[float] | None, default=None
        Optional normalization mean for denormalization.
    std : list[float] | None, default=None
        Optional normalization std for denormalization.
    max_images : int, default=8
        Maximum number of images to show.
    figsize : tuple[int, int], default=(16, 8)
        Figure size.

    Returns
    -------
    None
        Displays the figure.

    """
    num_images = min(max_images, images.size(0))

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
