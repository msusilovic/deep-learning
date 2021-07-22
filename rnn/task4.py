import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import load_data
import util
import numpy as np

from util import TRAIN_PATH, TEST_PATH, VALID_PATH, GLOVE_PATH


class RNNModel(nn.Module):
    def __init__(self, embedding_matrix, cell, rnn_sizes, fc_sizes, bidirectional = False):
        super().__init__()
        self.embedding_matrix = nn.Embedding.from_pretrained(torch.from_numpy(embedding_matrix), freeze=True)

        self.rnn_sizes = rnn_sizes

        self.rnn = list()
        self.fc = list()

        # two layer cells
        self.rnn.append(cell(rnn_sizes[0], rnn_sizes[1], 2, batch_first=False, dropout=0.2, bidirectional=bidirectional))
        self.rnn.append(cell(2 * rnn_sizes[1] if bidirectional else rnn_sizes[1], 150, 2, batch_first=False, dropout=0.2, bidirectional=bidirectional))
        self.fc.append(nn.Linear(2 * fc_sizes[0] if bidirectional else fc_sizes[0], fc_sizes[1]))
        self.fc.append(nn.Linear(fc_sizes[1], fc_sizes[2]))

    def forward(self, x):
        y = self.embedding_matrix(x).float()
        h = None
        y = torch.transpose(y, 0, 1)
        for rnn in self.rnn:
            y, h = rnn(y, h)

        y = y[-1]

        for fc in self.fc[:-1]:
            y = fc(y)
            y = torch.relu(y)

        return self.fc[-1](y)

    def infer(self, x):
        return torch.sigmoid(self.forward(x)).round().int()

    def parameters(self):
        params = list()
        for layer in self.rnn:
            params += list(layer.parameters())
        for layer in self.fc:
            params += list(layer.parameters())
        #params += self.embedding_matrix.parameters()
        return params


def train(model, data, optimizer, criterion):
    # sets model to train mode
    model.train()
    for batch_num, batch in enumerate(data):
        model.zero_grad()
        logits = model.forward(batch[0]).squeeze(-1)
        y = torch.tensor(list(batch[1])).float()

        loss = criterion(logits, y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)

        optimizer.step()
        optimizer.zero_grad()


if __name__ == "__main__":
    seed = 444444
    np.random.seed(seed)
    torch.manual_seed(seed)

    # create vocabulary from train_dataset
    word_frequencies, label_frequencies = load_data.get_frequencies(TRAIN_PATH)
    word_vocab = load_data.Vocab(frequencies=word_frequencies)
    label_vocab = load_data.Vocab(frequencies=label_frequencies, special_symbols=False)

    embedding_matrix = load_data.get_embedding_matrix(word_vocab, path=GLOVE_PATH)

    # get datasets
    train_dataset, valid_dataset, test_dataset = load_data.load_datasets(TRAIN_PATH, VALID_PATH, TEST_PATH, word_vocab, label_vocab)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=10, shuffle=True, collate_fn=load_data.pad_collate_fn)
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=32, shuffle=False, collate_fn=load_data.pad_collate_fn)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False, collate_fn=load_data.pad_collate_fn)

    rnn_sizes = [300, 150, 150]
    fc_sizes = [150, 150, 1]

    cells = {"RNN": nn.RNN, "GRU": nn.GRU, "LSTM": nn.LSTM}
    for name, cell in cells.items():
        print("\nCell: ", name)
        model = RNNModel(embedding_matrix, cell, rnn_sizes, fc_sizes, bidirectional=True)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        for epoch in range(1, 5):
            train(model, train_dataloader, optimizer, criterion)
            util.evaluate(model, valid_dataloader, criterion)

        print("\nTest dataset:")
        util.evaluate(model, test_dataloader, criterion)
