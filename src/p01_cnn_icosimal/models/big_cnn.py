import torch
import torch.nn as nn
import torch.nn.functional as F

class BigCNN(nn.Module):
    """A big-sized CNN model for image classification with two additional convolutional layers.
    Additionally it includes batch normalization"""

    def __init__(self, num_classes: int = 10, input_channels: int = 3, image_size: int = 128) -> None:
        """
        Initialize the BigCNN model.

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
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)


        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # For 64x64 inputs:
        # 64 -> 32 -> 16 -> 8 -> 4 after four MaxPool layers

        fc_input_size = 256 * (image_size // 16) * (image_size // 16)

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
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        # Flatten and fully connected layers
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
    
    