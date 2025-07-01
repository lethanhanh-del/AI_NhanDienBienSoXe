import torch
import torch.nn as nn

class CharacterCNN(nn.Module):
    def __init__(self, num_classes=36):
        super(CharacterCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, kernel_size=3, padding=1)  # Tăng filter
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(512)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(512 * 4 * 4, 1024)  # Tăng kích thước lớp FC
        self.bn_fc1 = nn.BatchNorm1d(1024)  # Thêm BatchNorm
        self.fc2 = nn.Linear(1024, 256)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 512 * 4 * 4)
        x = self.dropout(self.relu(self.bn_fc1(self.fc1(x))))
        x = self.dropout(self.relu(self.bn_fc2(self.fc2(x))))
        x = self.fc3(x)
        return x