import torch
import torch.nn as nn
import torch.nn.functional as F

class MNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)  # 1-image channel, 32-output channel, 5-kernel
        self.conv2 = nn.Conv2d(32, 64, 3)  # 10-input channel, 20-output channel, 3-kernel
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 10 * 10, 500)  # input channel, output channel
        self.fc2 = nn.Linear(500, 10)  # input channel, answer vector

    def forward(self, x):
        x = self.conv1(x)  # input: batch*1*28*28, output: batch*32*24*24
        x = F.leaky_relu(x)
        x = F.max_pool2d(x, 2, 2)  # input: batch*32*24*24, output: batch*32*12*12

        x = self.conv2(x)  # input: batch*32*12*12, output: batch*64*10*10
        x = F.leaky_relu(x)

        x = self.dropout(x)

        x = torch.flatten(x, 1)

        x = self.fc1(x)  # input: batch*6400, output: batch*500
        x = F.leaky_relu(x)

        x = self.fc2(x)  # input: batch*500, output: batch*10

        output = F.log_softmax(x, dim=1)  # calculate max prob. of all num
        return output