import os.path
import time

import numpy as np
import torch
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader

from models.LSTM import *

# %%
# parameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 96
EPOCHS = 16
SAVEPATH = 'models/LSTM.pth'

# %%
print('loading datas ...')
datas = np.load('tang.npz', allow_pickle=True)
data = datas['data']
ix2word = datas['ix2word'].item()
word2ix = datas['word2ix'].item()
# numpy to tensor
data = torch.from_numpy(data)
train_loader = DataLoader(dataset=data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

# %%
model = Poem(len(word2ix), embedding_dim=128, hidden_dim=256)
if os.path.exists(SAVEPATH):
    model = torch.load(SAVEPATH).to(DEVICE)
else:
    model = model.to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=3e-6)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4)
start_time = time.time()


# %%
def train(epoch):
    model.train()
    Loss = 0.0
    print('learning rate %.6f' % (optimizer.param_groups[0]['lr']))
    for batch_idx, data in enumerate(train_loader):
        data = data.long().transpose(1, 0).contiguous()
        data = data.to(DEVICE)
        optimizer.zero_grad()
        input, target = data[:-1, :], data[1:, :]
        output, _ = model(input)
        loss = criterion(output, target.view(-1))
        loss.backward()
        optimizer.step()
        Loss += loss

        if (batch_idx > 0) and (batch_idx % 100 == 0):
            print('training time %.6f, epoch %d, solved %d, current loss %.6f' % (
                time.time() - start_time, epoch, batch_idx, Loss / (batch_idx + 1.0)))

        # if (batch_idx + 1) % 200 == 0:
        #     break
    scheduler.step()


# %%
def generate(start_words, max_gen_len, prefix_words=None):
    results = list(start_words)
    start_word_len = len(start_words)

    input = torch.Tensor([word2ix['<START>']]).view(1, 1).long()
    input = input.to(DEVICE)
    hidden = None

    if prefix_words:
        for word in prefix_words:
            output, hidden = model(input, hidden)
            input = Variable(input.data.new([word2ix[word]])).view(1, 1)

    # Variable表示含有梯度的变量
    # .data表示tensor中的数值 且修改后tensor中也被修改 不可用来更新梯度
    for i in range(max_gen_len):
        output, hidden = model(input, hidden)
        if i < start_word_len:
            w = results[i]
            input = input.detach().new([word2ix[w]]).view(1, 1)
        else:
            top_index = output.data[0].topk(1)[1][0].item()
            w = ix2word[top_index]
            results.append(w)
            input = input.data.new([top_index]).view(1, 1)
        if w == '<EOP>':
            del results[-1]
            break
    return results


# %%
def test():
    start_words = '烟笼寒水月笼沙'
    max_gen_len = 32
    prefix_words = None
    results = generate(start_words, max_gen_len)
    poetry = ''
    for word in results:
        poetry += word
        if word == '。' or word == '！':
            poetry += '\n'
    print(poetry)


# %%
if __name__ == '__main__':
    # pass
    test()
    # for epoch in range(EPOCHS):
    #     train(epoch)
    #     torch.save(model, SAVEPATH)
