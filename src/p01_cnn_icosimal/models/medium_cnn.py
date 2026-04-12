import torch
import torch.nn as nn
import torch.nn.functional as F

class MediumCNN(nn.Module):
    """A medium-sized CNN model for image classification with an additional convolutional layer
    and increased amount of filters. Additionally it includes batch normalization"""

    def __init__(self, num_classes: int = 10, input_channels: int = 3, image_size: int = 128) -> None:
        """
        Initialize the MediumCNN model.

        Args:
            num_classes: Number of output classes.
            input_channels: Number of input channels.
            image_size: Size of the input images.

        """
        super().__init__()

        self.image_size = image_size

        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)


        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # For 64x64 inputs:
        # 64 -> 32 -> 16 -> 8 after three MaxPool layers

        fc_input_size = 128 * (image_size // 8) * (image_size // 8)

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
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        # Flatten and fully connected layers
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
    
    