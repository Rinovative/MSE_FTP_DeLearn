"""Baseline CNN model for image classification."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class TestCNN(nn.Module):
    """A test CNN model for image classification."""

    def __init__(self, num_classes: int = 10, input_channels: int = 3) -> None:
        """
        Initialize the TestCNN model.

        Args:
            num_classes: Number of output classes.
            input_channels: Number of input channels.

        """
        super().__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # Pooling and batch normalization
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

        # For 128x128 inputs:
        # 128 -> 64 -> 32 -> 16 after the three MaxPool layers
        # Then fixed average pooling: 16x16 -> 4x4
        self.final_pool = nn.AvgPool2d(kernel_size=4, stride=4)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor.

        Returns:
            Output logits.

        """
        # Conv block 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))

        # Conv block 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))

        # Conv block 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        # Fixed pooling, flatten, and fully connected layers
        x = self.final_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)
