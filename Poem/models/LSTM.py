import torch
import torch.nn as nn

# %%
LSTMLAYERS = 3


# %%
class Poem(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(Poem, self).__init__()
        self.hidden_dim = hidden_dim
        # 词向量 词表大小 * 向量维度
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # 核心LSTM层
        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim, num_layers=LSTMLAYERS)

        # 分类过程
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 2048),
            nn.ReLU(),
            nn.Linear(2048, vocab_size)
        )

    def forward(self, input, hidden=None):
        seq_len, batch_size = input.size()
        if hidden is None:
            h_0 = input.data.new(LSTMLAYERS, batch_size, self.hidden_dim).fill_(0).float()
            c_0 = input.data.new(LSTMLAYERS, batch_size, self.hidden_dim).fill_(0).float()
        else:
            h_0, c_0 = hidden

        # input: length * batch
        # output: length * batch * 向量维度
        # hidden: length * batch * hidden_dim
        embedings = self.embedding(input)
        output, hidden = self.lstm(embedings, (h_0, c_0))
        output = self.classifier(output.view(seq_len * batch_size, -1))

        return output, hidden
