from collections import Counter
from dataclasses import dataclass

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataset import T_co
from torch.nn.utils.rnn import pad_sequence

from util import TRAIN_PATH

import torch
import numpy as np

PADDING = "<PAD>"
UNKNOWN = "<UNK>"


@dataclass
class Instance:
    instance_text: list
    instance_label: str


class NLPDataset(Dataset):
    def __init__(self, path, text_vocab, label_vocab):
        self.text_vocab = text_vocab
        self.label_vocab = label_vocab

        self.instances = []
        lines = open(path, 'r').readlines()
        for line in lines:
            parts = line.rstrip().split(', ')
            words = parts[0].split(' ')
            self.instances.append(Instance(words, parts[1]))

    def __getitem__(self, index) -> T_co:
        instance = self.instances[index]
        return torch.tensor(self.text_vocab.encode(instance.instance_text)), torch.tensor(self.label_vocab.encode(instance.instance_label))

    def __len__(self):
        return len(self.instances)


class Vocab:
    def __init__(self, frequencies, max_size=-1, min_freq=1, special_symbols=True):
        frequencies = {k: v for (k, v) in frequencies.items() if v >= min_freq}.items()
        frequencies = sorted(frequencies, key=lambda x: x[1], reverse=True)
        self.special_symbols = special_symbols
        self.itos = dict()
        self.stoi = dict()

        if max_size >= 0:
            frequencies = frequencies[:max_size-2]

        if special_symbols:
            self.itos[0] = PADDING
            self.itos[1] = UNKNOWN
            self.stoi[PADDING] = 0
            self.stoi[UNKNOWN] = 1

        for i, (k, _) in enumerate(frequencies, start=2 if special_symbols else 0):
            self.itos[i] = k
            self.stoi[k] = i

    # returns list of indices for tokens from stoi dictionary, or index 1 if token is not in dictionary
    def encode(self, tokens):
        if isinstance(tokens, list):
            return [self.stoi.get(token, (self.stoi.get('<UNK>') if self.special_symbols else None)) for token in tokens]

        return self.stoi.get(tokens, (self.stoi.get('<UNK>') if self.special_symbols else None))

    # returns list of tokens for indices from itos dictionary
    def decode(self, ids):
        if isinstance(ids, list):
            return [self.stoi.get(token) for token in ids]

        return self.stoi.get(ids)


def get_embedding_matrix(vocab, path=None, dimension=300):
    words = vocab.itos
    vector_dict = dict()

    # read vector representations from file
    if path:
        lines = open(path, 'r').readlines()
        for line in lines:
            parts = line.split()
            vector_dict[parts[0]] = list(map(float, parts[1:]))

    # init all values with Gaussian distribution
    embedding_matrix = np.random.randn(len(words), 300)

    for (index, word) in words.items():
        if word == PADDING:
            embedding_matrix[index] = np.zeros(dimension)
        elif word in vector_dict:
            embedding_matrix[index] = vector_dict[word]

    return embedding_matrix


def pad_collate_fn(batch, pad_index=0):
    """
    Arguments:
        batch:
            list of Instances returned by `Dataset.__getitem__`.
        pad_index:
            index of padding symbol
    Returns:
      A tensor representing the input batch.
    """

    texts, labels = zip(*batch)
    lengths = [len(text) for text in texts]
    texts = pad_sequence(texts, batch_first=True, padding_value=pad_index)

    return texts, labels, lengths


def get_frequencies(path):
    lines = open(path, 'r').readlines()
    word_frequencies = Counter()
    label_frequencies = Counter()
    for line in lines:
        # split text from label
        parts = line.rstrip().split(', ')
        # split words
        words = parts[0].split(' ')
        label_frequencies[parts[1]] += 1
        for word in words:
            word_frequencies[word] += 1

    return word_frequencies, label_frequencies


def load_datasets(train_path, valid_path, test_path, word_vocab, label_vocab):
    train_dataset = NLPDataset(train_path, word_vocab, label_vocab)
    valid_dataset = NLPDataset(valid_path, word_vocab, label_vocab)
    test_dataset = NLPDataset(test_path, word_vocab, label_vocab)

    return train_dataset, valid_dataset, test_dataset


if __name__ == "__main__":
    batch_size = 3
    shuffle = False

    word_frequencies, label_frequencies = get_frequencies(TRAIN_PATH)
    word_vocab = Vocab(word_frequencies, special_symbols=True)
    label_vocab = Vocab(label_frequencies, special_symbols=False)

    train_dataset = NLPDataset(TRAIN_PATH, word_vocab, label_vocab)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                                  shuffle=shuffle, collate_fn=pad_collate_fn)

    texts, labels, lengths = next(iter(train_dataloader))
    print(f"Texts: {texts}")
    print(f"Labels: {labels}")
    print(f"Lengths: {lengths}")