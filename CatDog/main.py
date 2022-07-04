import os
import random
import shutil

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torchvision.transforms
from torchvision import datasets
from torch.utils.data import DataLoader

from models.CNN import CNN

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# %%
TestPath = 'data/test1'
TrainPath = 'data/train'

DatasetTrain = 'dataset/train'
DatasetValidation = 'dataset/validation'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(DEVICE)
BATCH_SIZE = 128
EPOCHS = 16
LEARNING_RATE = 1e-3
SAVEPATH = 'models/CNN.pth'


# %%
def moveFiles():
    files = os.listdir(TrainPath)
    id = 0
    ct = 0
    for file in files:
        src = os.path.join(TrainPath, file)
        num = random.random()
        if num >= 0.8:
            ct += 1
            if 'cat' in str(file):
                tar = os.path.join(DatasetValidation, 'cats', file)
            else:
                tar = os.path.join(DatasetValidation, 'dogs', file)
        else:
            if 'cat' in str(file):
                tar = os.path.join(DatasetTrain, 'cats', file)
            else:
                tar = os.path.join(DatasetTrain, 'dogs', file)
        shutil.copyfile(src, tar)
        id += 1
        if id % 1000 == 0:
            print('%d files moved, %d train, %d validation' % (id, id - ct, ct))


# %%
def checkFiles():
    print(len(os.listdir(os.path.join(DatasetTrain, 'cats'))))
    print(len(os.listdir(os.path.join(DatasetTrain, 'dogs'))))
    print(len(os.listdir(os.path.join(DatasetValidation, 'cats'))))
    print(len(os.listdir(os.path.join(DatasetValidation, 'dogs'))))


# %%
transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(364),
    torchvision.transforms.CenterCrop(364),
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
train_dataset = datasets.ImageFolder(root=DatasetTrain, transform=transforms)
test_dataset = datasets.ImageFolder(root=DatasetValidation, transform=transforms)

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)

print("Loading Data Finished")
# %%
if os.path.exists(SAVEPATH):
    model = torch.load(SAVEPATH).to(DEVICE)
else:
    model = CNN().to(DEVICE)
optimizer = optim.Adam(model.parameters())


# %%
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    Loss = 0.0
    for batch_index, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        pred = output.softmax(dim=1)
        loss.backward()
        Loss += loss.item()
        optimizer.step()
        if  (batch_index > 0) and(batch_index % 25 == 0):
            # f = open(SAVEPATH, 'w')
            # f.close()
            print("train epoch %d, batch %d, current loss %.6f" % (epoch, batch_index, Loss / (1.0 + batch_index)))
    torch.save(model, SAVEPATH)


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
if __name__ == '__main__':
    for epoch in range(0, EPOCHS):
        train(model, DEVICE, train_loader, optimizer, epoch)
        evaluate(model, DEVICE, test_loader)
