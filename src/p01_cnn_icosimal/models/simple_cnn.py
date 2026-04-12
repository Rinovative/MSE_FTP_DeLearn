import torch
import torch.nn as nn
import torch.nn.functional as f

class SimpleCNN(nn.Module):
    """A simple CNN model for image classification."""

    def __init__(self, num_classes: int = 10, input_channels: int = 3, image_size: int = 128) -> None:
        """
        Initialize the SimpleCNN model.

        Args:
            num_classes: Number of output classes.
            input_channels: Number of input channels.
            image_size: Size of the input images.

        """
        super().__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # For 128x128 inputs:
        # 128 -> 64 -> 32 after two MaxPool layers

        fc_input_size = 32 * (image_size // 4) * (image_size // 4)

        # Fully connected layers
        self.fc1 = nn.Linear(fc_input_size, 128)
        self.fc2 = nn.Linear(128, num_classes)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor.

        Returns:
            Output logits.

        """
        x = self.pool(f.relu(self.conv1(x)))
        x = self.pool(f.relu(self.conv2(x)))

        # Flatten and fully connected layers
        x = torch.flatten(x, start_dim=1)
        x = f.relu(self.fc1(x))
        x = self.fc2(x)

        return x
    
    def get_model_config(self) -> dict:
        return {
            "num_classes": self.fc2.out_features,
            "input_channels": self.conv1.in_channels,
            "image_size": self.image_size,
        }