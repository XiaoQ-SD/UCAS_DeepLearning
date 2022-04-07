import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 5)
        self.conv4 = nn.Conv2d(64, 128, 5)
        self.droupout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 12 * 12, 512)
        self.fc2 = nn.Linear(512, 32)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x):
        # input: batch*3*512*512
        x = self.conv1(x)
        # output: batch*16*508*508
        x = F.leaky_relu(x)
        x = F.max_pool2d(x, 2, 2)
        # output: batch*16*254*254

        x = self.conv2(x)
        # output: batch*32*252*252
        x = F.leaky_relu(x)
        x = F.max_pool2d(x, 3, 3)
        # output: batch*32*84*84

        x = self.conv3(x)
        # output: batch*64*80*80
        x = F.leaky_relu(x)
        x = F.max_pool2d(x, 2, 2)
        # output: batch*64*40*40

        x = self.conv4(x)
        # output: batch*128*36*36
        x = F.leaky_relu(x)
        x = F.max_pool2d(x, 3, 3)
        # output: batch*128*12*12

        x = self.dropout(x)
        x = torch.flatten(x, 1)
        # output: batch*18432

        x = self.fc1(x)
        x = F.leaky_relu(x)
        # output: batch*512

        x = self.fc2(x)
        x = F.leaky_relu(x)
        # output: batch*32

        x = self.fc3(x)
        x = F.leaky_relu(x)
        # output: batch*2

        # calculate max prob. of all num
        output = F.log_softmax(x, dim=1)
        return output
