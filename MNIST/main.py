# %%
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torchvision.transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from models.MNIST import MNIST

# %%
# paramaters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
EPOCHS = 16
LEARNING_RATE = 1e-3

# %%
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.1307,), (0.3081,))
])
# %%
print("Loading Datas ...")
train_dataset = datasets.MNIST(root='data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='data', train=False, transform=transform, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)
print("Loading Datas Finished")

print("train datas of %d" % len(train_loader))
print("test datas of %d" % len(test_loader))
# %%
model = MNIST().to(DEVICE)
optimizer = optim.Adam(model.parameters())


# %%
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    Loss = 0.0
    for batch_index, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()  # 梯度初始化为0
        output = model(data)
        loss = F.cross_entropy(output, target)  # 损失
        pred = output.softmax(dim=1)  # 概率值的最大下标
        loss.backward()
        Loss += loss.item()
        optimizer.step()
        if (batch_index > 0) and (batch_index % 100 == 0):
            print("train epoch %d, batch %d, current loss %.6f" % (epoch, batch_index, (Loss / (1.0 + batch_index))))


# %%
def evaluate(model, device, test_loader):
    model.eval()
    correct = 0.0
    test_loss = 0.0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        print("test average loss %.4f, accuracy %.4f" % (test_loss, 100.0 * correct / len(test_loader.dataset)))


# %%
for epoch in range(0, EPOCHS):
    train(model, DEVICE, train_loader, optimizer, epoch)
    evaluate(model, DEVICE, test_loader)
