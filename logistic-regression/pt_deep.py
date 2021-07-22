import data
import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD
import matplotlib.pyplot as plt


class PTDeep(nn.Module):
    def __init__(self, sizes, activation):
        super().__init__()

        torch.manual_seed(1000)
        ws = []
        bs = []
        for i in range(1, len(sizes)):
            ws.append(nn.Parameter(torch.randn(sizes[i-1], sizes[i])))
            bs.append(nn.Parameter(torch.zeros([sizes[i]])))

        self.weights = nn.ParameterList(ws)
        self.biases = nn.ParameterList(bs)

        self.activation = activation

    def forward(self, X):
        h = X.clone().double()
        for i in range(len(self.biases)):
            h = h.mm(self.weights[i].double()) + self.biases[i]
            h = self.activation(h)

        probs = torch.softmax(h, dim=1)

        return probs

    def get_loss(self, X, Yoh_):
        probs = self.forward(X)
        logprobs = torch.log(probs) * Yoh_
        logsum = torch.sum(logprobs, dim=1)
        logprobs_mean = torch.mean(logsum)

        return -logprobs_mean

    def count_params(self):
        return np.sum(p.numel() for p in self.parameters())


def train(model, X, Yoh_, param_niter, param_delta=0.2, param_reg=0.001, verbose=True):
    optimizer = SGD(model.parameters(), lr=param_delta, weight_decay=0.01)

    for i in range(param_niter):
        loss = model.get_loss(X, Yoh_)
        loss.backward()

        if verbose and i % 10 == 0:
            print("iteration {}: loss {}".format(i, loss))

        optimizer.step()
        optimizer.zero_grad()


def eval(model, X):
    """Arguments:
       - model: type: PTLogreg
       - X: actual datapoints [NxD], type: np.array
       Returns: predicted class probabilites [NxC], type: np.array
    """
    # ulaz je potrebno pretvoriti u torch.Tensor
    # izlaze je potrebno pretvoriti u numpy.array
    # koristite torch.Tensor.detach() i torch.Tensor.numpy()

    return np.argmax(model.forward(torch.from_numpy(X)).detach().numpy(), axis=1)


if __name__ == "__main__":
    # inicijaliziraj generatore slučajnih brojeva
    np.random.seed(100)

    # instanciraj podatke X i labele Yoh_
    X, Y_ = data.sample_gmm_2d(6, 2, 10)
    x = torch.from_numpy(X)
    Yoh_ = torch.from_numpy(data.class_to_onehot(Y_))

    # definiraj model:
    ptd = PTDeep([2, 10, 10, 2], torch.sigmoid)
    print(f'broj parametara: {ptd.count_params()}')
    # nauči parametre (X i Yoh_ moraju biti tipa torch.Tensor):
    train(ptd, x, Yoh_, 10000)

    # dohvati vjerojatnosti na skupu za učenje
    Y = eval(ptd, X)

    # ispiši performansu (preciznost i odziv po razredima)
    accuracy, precision, confusion_matrix = data.eval_perf_multi(Y, Y_)
    print('Accuracy: ', accuracy)
    print('Precision: ', precision)
    print('Confusion matrix:\n', confusion_matrix)

    # iscrtaj rezultate, decizijsku plohu
    bbox = (np.min(X, axis=0), np.max(X, axis=0))
    decision = lambda X: eval(ptd, X)
    data.graph_surface(decision, bbox, offset=0.5)

    # graph the data points
    data.graph_data(X, Y_, Y, special=[])
    plt.show()