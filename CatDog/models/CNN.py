import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 20, 5)
        self.conv2 = nn.Conv2d(20, 40, 5)
        self.conv3 = nn.Conv2d(40, 60, 3)
        self.conv4 = nn.Conv2d(60, 80, 3)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(13520, 256)
        self.fc2 = nn.Linear(256, 32)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x):
        # input: batch*3*364*364
        x = self.conv1(x)
        # output: batch*20*360*360
        x = F.leaky_relu(x)
        x = F.max_pool2d(x, 3, 3)
        # output: batch*20*120*120

        x = self.conv2(x)
        # output: batch*40*116*116
        x = F.leaky_relu(x)
        x = F.max_pool2d(x, 2, 2)
        # output: batch*40*58*58

        x = self.conv3(x)
        # output: batch*60*56*56
        x = F.leaky_relu(x)
        x = F.max_pool2d(x, 2, 2)
        # output: batch*60*28*28

        x = self.conv4(x)
        # output: batch*80*26*26
        x = F.leaky_relu(x)
        x = F.max_pool2d(x, 2, 2)
        # output: batch*80*13*13

        x = self.dropout(x)
        x = torch.flatten(x, 1)
        # output: batch*13520

        x = self.fc1(x)
        x = F.leaky_relu(x)
        # output: batch*256

        x = self.fc2(x)
        x = F.leaky_relu(x)
        # output: batch*32

        x = self.fc3(x)
        x = F.leaky_relu(x)
        # output: batch*2

        # calculate max prob. of all num
        output = F.log_softmax(x, dim=1)
        return output
