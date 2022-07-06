import os.path
import time

import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import *
from utils import *
from model.TextCNN import *

word2id = build_word2id()
# 构建训练集+验证集的词汇表
# print(word2id)

word2vec = build_word2vec('Dataset/wiki_word2vec_50.bin', word2id)
assert word2vec.shape == (58954, 50)
# 基于构建好的word2vec构建训练语料中所含词语的word2vec
# print(word2vec)

# 读取数据集 {id, id, ...} -> {0|1}
# print('train set: ')
train_contents, train_labels = load_corpus('Dataset/train.txt', word2id, max_sen_len=50)
# print('validation set: ')
val_contents, val_labels = load_corpus('Dataset/validation.txt', word2id, max_sen_len=50)
# print('test set: ')
test_contents, test_labels = load_corpus('Dataset/test.txt', word2id, max_sen_len=50)


class CONFIG():
    update_w2v = True  # 是否在训练中更新w2v
    vocab_size = 58954  # 词汇量，与word2id中的词汇量一致
    n_class = 2  # 分类数：分别为pos和neg
    embedding_dim = 50  # 词向量维度
    drop_keep_prob = 0.5  # dropout层，参数keep的比例
    kernel_num = 64  # 卷积层filter的数量
    kernel_size = [3, 4, 5]  # 卷积核的尺寸
    pretrained_embed = word2vec  # 预训练的词嵌入模型


config = CONFIG()  # 配置模型参数
learning_rate = 0.001  # 学习率
BATCH_SIZE = 128  # 训练批量
EPOCHS = 16  # 训练轮数
SAVEPATH = 'model/TextCNN.pth'  # 预训练模型路径
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_dataset = TensorDataset(torch.from_numpy(train_contents).type(torch.float),
                              torch.from_numpy(train_labels).type(torch.long))
train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

val_dataset = TensorDataset(torch.from_numpy(val_contents).type(torch.float),
                            torch.from_numpy(val_labels).type(torch.long))
val_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

if os.path.exists(SAVEPATH):
    model = torch.load(SAVEPATH)
else:
    model = TextCNN(config)
model = model.to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()
scheduler = StepLR(optimizer, step_size=8, gamma=0.33)
start_time = time.time()

train_losses = []
train_acces = []
val_losses = []
val_acces = []


def train(dataloader, epoch):
    # 定义训练过程
    Loss, train_acc = 0.0, 0.0
    count, correct = 0, 0
    print('learning rate %.6f' % (optimizer.param_groups[0]['lr']))
    for batch_idx, (x, y) in enumerate(dataloader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        Loss += loss.item()
        correct += (output.argmax(1) == y).float().sum().item()
        count += len(x)

        if (batch_idx > 0) and (batch_idx % 50 == 0):
            print('training time %.6f, epoch %d, solved %d, current loss %.6f' % (
                time.time() - start_time, epoch, batch_idx, Loss / (batch_idx + 1.0)))

    Loss *= BATCH_SIZE
    Loss /= len(dataloader.dataset)
    train_acc = correct / count
    print('epoch %d finished, average loss %.6f, accuracy %.4f' % (epoch, Loss, train_acc))
    scheduler.step()


def validation(dataloader, epoch):
    model.eval()
    # 验证过程
    val_loss, val_acc = 0.0, 0.0
    count, correct = 0, 0
    for _, (x, y) in enumerate(dataloader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        output = model(x)
        loss = criterion(output, y)
        val_loss += loss.item()
        correct += (output.argmax(1) == y).float().sum().item()
        count += len(x)

    val_loss *= BATCH_SIZE
    val_loss /= len(dataloader.dataset)
    val_acc = correct / count
    print('validate %d finished, loss %.6f, accuracy %.4f' % (epoch, val_loss, val_acc))


test_dataset = TensorDataset(torch.from_numpy(test_contents).type(torch.float),
                             torch.from_numpy(test_labels).type(torch.long))
test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)


def test(dataloader):
    print('start testing')
    model.eval()
    model.to(DEVICE)

    # 测试过程
    cnt, correct = 0, 0
    for _, (x, y) in enumerate(dataloader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        output = model(x)
        correct += (output.argmax(1) == y).float().sum().item()
        cnt += len(x)

    # 打印准确率
    print('test accuracy %.4f' % (correct / cnt))


if __name__ == '__main__':
    test(test_dataloader)
    # for epoch in range(1, EPOCHS + 1):
    #     train(train_dataloader, epoch)
    #     validation(val_dataloader, epoch)
    #     torch.save(model, SAVEPATH)
    #     test(test_dataloader)
