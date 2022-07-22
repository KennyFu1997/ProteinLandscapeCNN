import torch
import torch.nn as nn


class ProtCNN(nn.Module):
    def __init__(self, num_classes=21):
        # (4, 20, 20, 20)
        super(ProtCNN, self).__init__()
        # torch.nn.Conv3d(in_channels, out_channels, kernel_size)
        self.features = nn.Sequential(
            nn.Conv3d(4, 100, 3),
            nn.ReLU(),
            nn.Conv3d(100, 200, 3),
            nn.ReLU(),
            nn.MaxPool3d(2),
            
            nn.Conv3d(200, 200, 2),
            nn.ReLU(),
            nn.Conv3d(200, 400, 2),
            nn.ReLU(),
            nn.MaxPool3d(2)
        )
        self.clf = nn.Sequential(
            nn.Flatten(),
            nn.Linear(10800, 1000),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1000, 100),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(100, num_classes)),
            nn.Softmax(),
        
    def forward(self, input):
        x = self.features(input)
        return self.clf(x)

