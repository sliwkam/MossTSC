"""
This module defines four deep learning architectures for classification tasks:
    - LeNet2D: A 2D convolutional neural network with dropout
    - LeNet3D: A 3D convolutional neural network with dropout
    - AlexNet2D: A 2D version of AlexNet with dropout
    - AlexNet3D: A 3D version of AlexNet with dropout
Each model uses adaptive pooling and calculates its fully connected layer input size dynamically based on
the provided input dimension (alpha).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet2D(nn.Module):
    """
    LeNet2D architecture with dropout for 2D inputs.

    Args:
        num_classes (int): number of output classes
        alpha (int): size of the input matrix (assumed square: alpha x alpha)
        dropout_rate (float, optional): dropout probability
    """
    def __init__(self, num_classes, alpha, dropout_rate=0.5):
        super(LeNet2D, self).__init__()
        self.alpha = alpha
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Convolutional layers with padding to preserve spatial dimensions.
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, padding=2)
        
        # Calculate the input size for the fully connected layers dynamically.
        fc_input_size = self._get_fc_input_size()
        
        self.fc1 = nn.Linear(fc_input_size, 120)
        self.dropout1 = nn.Dropout(p=self.dropout_rate)
        self.fc2 = nn.Linear(120, 84)
        self.dropout2 = nn.Dropout(p=self.dropout_rate)
        self.fc3 = nn.Linear(84, num_classes)

    def _get_fc_input_size(self):
        """
        Compute the number of features output by the convolutional layers.

        Returns:
            int: the flattened feature size
        """
        # Create a dummy input with shape (1, 1, alpha, alpha).
        x = torch.zeros(1, 1, self.alpha, self.alpha)
        # Apply convolution and pooling operations as in forward().
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=2, stride=2)
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=2, stride=2)
        return x.numel()

    def forward(self, x):
        """
        Forward pass for LeNet2D.

        Args:
            x (torch.Tensor): input tensor of shape (batch_size, 1, alpha, alpha)

        Returns:
            torch.Tensor: output logits of shape (batch_size, num_classes)
        """
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=2, stride=2)
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=2, stride=2)
        x = x.view(x.size(0), -1)  # Flatten the tensor.
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


class LeNet3D(nn.Module):
    """
    LeNet3D architecture with dropout for 3D inputs.

    Args:
        num_classes (int): number of output classes
        alpha (int): size of the input volume (assumed cube: alpha x alpha x alpha)
        dropout_rate (float, optional): dropout probability
    """
    def __init__(self, num_classes, alpha, dropout_rate=0.5):
        super(LeNet3D, self).__init__()
        self.alpha = alpha
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # 3D Convolutional layers with padding to preserve spatial dimensions.
        self.conv1 = nn.Conv3d(1, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv3d(6, 16, kernel_size=5, padding=2)
        
        fc_input_size = self._get_fc_input_size()
        
        self.fc1 = nn.Linear(fc_input_size, 120)
        self.dropout1 = nn.Dropout(p=self.dropout_rate)
        self.fc2 = nn.Linear(120, 84)
        self.dropout2 = nn.Dropout(p=self.dropout_rate)
        self.fc3 = nn.Linear(84, num_classes)

    def _get_fc_input_size(self):
        """
        Compute the flattened feature size after 3D convolutions and pooling.

        Returns:
            int: the number of features to be fed into the fully connected layer
        """
        x = torch.zeros(1, 1, self.alpha, self.alpha, self.alpha)
        x = F.max_pool3d(F.relu(self.conv1(x)), kernel_size=2, stride=2)
        x = F.max_pool3d(F.relu(self.conv2(x)), kernel_size=2, stride=2)
        return x.numel()

    def forward(self, x):
        """
        Forward pass for LeNet3D.

        Args:
            x (torch.Tensor): input tensor of shape (batch_size, 1, alpha, alpha, alpha)

        Returns:
            torch.Tensor: output logits of shape (batch_size, num_classes)
        """
        x = F.max_pool3d(F.relu(self.conv1(x)), kernel_size=2, stride=2)
        x = F.max_pool3d(F.relu(self.conv2(x)), kernel_size=2, stride=2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


class AlexNet2D(nn.Module):
    """
    AlexNet2D architecture with dropout for 2D inputs.

    Args:
        num_classes (int): number of output classes
        alpha (int): size of the input image (assumed square: alpha x alpha)
        dropout_rate (float): dropout probability
    """
    def __init__(self, num_classes, alpha, dropout_rate):
        super(AlexNet2D, self).__init__()
        self.alpha = alpha
        
        # Define convolutional layers.
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(p=dropout_rate)
        
        # Adaptive average pooling layer.
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Dynamically compute the fully connected input size.
        self.fc_input_size = self._get_fc_input_size()
        self.fc1 = nn.Linear(self.fc_input_size, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)

    def _get_fc_input_size(self):
        """
        Calculate the flattened feature size after convolutions and pooling.

        Returns:
            int: the number of input features for the fully connected layers
        """
        x = torch.zeros(1, 1, self.alpha, self.alpha)
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=2, stride=2)
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=2, stride=2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.avgpool(x)
        return x.numel()

    def forward(self, x):
        """
        Forward pass for AlexNet2D.

        Args:
            x (torch.Tensor): input tensor of shape (batch_size, 1, alpha, alpha)

        Returns:
            torch.Tensor: output logits of shape (batch_size, num_classes)
        """
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class AlexNet3D(nn.Module):
    """
    AlexNet3D architecture with dropout for 3D inputs.

    Args:
        num_classes (int): number of output classes
        alpha (int): size of the input volume (assumed cube: alpha x alpha x alpha)
        dropout_rate (float): dropout probability
    """
    def __init__(self, num_classes, alpha, dropout_rate):
        super(AlexNet3D, self).__init__()
        self.alpha = alpha
        
        # Define 3D convolutional layers.
        self.conv1 = nn.Conv3d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(64, 192, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(192, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv3d(384, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv3d(256, 256, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(p=dropout_rate)
        
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Dynamically compute the fully connected input size.
        self.fc_input_size = self._get_fc_input_size()
        self.fc1 = nn.Linear(self.fc_input_size, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)

    def _get_fc_input_size(self):
        """
        Calculate the flattened feature size after 3D convolutions and pooling.

        Returns:
            int: the number of features to be fed into the fully connected layers
        """
        x = torch.zeros(1, 1, self.alpha, self.alpha, self.alpha)
        x = F.max_pool3d(F.relu(self.conv1(x)), kernel_size=2, stride=2)
        x = F.max_pool3d(F.relu(self.conv2(x)), kernel_size=2, stride=2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.avgpool(x)
        return x.numel()

    def forward(self, x):
        """
        Forward pass for AlexNet3D.

        Args:
            x (torch.Tensor): input tensor of shape (batch_size, 1, alpha, alpha, alpha)

        Returns:
            torch.Tensor: output logits of shape (batch_size, num_classes)
        """
        x = F.relu(self.conv1(x))
        x = F.max_pool3d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool3d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
