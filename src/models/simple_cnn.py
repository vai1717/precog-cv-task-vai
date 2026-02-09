import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """
    Standard CNN for 28x28 RGB images.
    
    Architecture:
    - Conv1: 3 -> 32 channels, 3x3, padding=1
    - Conv2: 32 -> 64 channels, 3x3, padding=1
    - Conv3: 64 -> 128 channels, 3x3, padding=1
    - Global Average Pooling
    - FC: 128 -> 10
    """
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2) # 28 -> 14
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2) # 14 -> 7
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2) # 7 -> 3
        )
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
