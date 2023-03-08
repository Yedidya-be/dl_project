import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, out_dim = 64):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(5, 16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, out_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.to(torch.float16)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = x.view(-1, 32 * 32 * 32)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
