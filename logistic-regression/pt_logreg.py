import data
import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD
import matplotlib.pyplot as plt


class PTLogreg(nn.Module):
    def __init__(self, D, C):
        """Arguments:
           - D: dimensions of each datapoint
           - C: number of classes
        """
        super().__init__()

        self.W = nn.Parameter(torch.randn(C, D), requires_grad=True)
        self.b = nn.Parameter(torch.zeros([C]), requires_grad=True)

    def forward(self, X):
        scores = X.mm(self.W.t().double()) + self.b
        probs = torch.softmax(scores, dim=1)

        return probs

    def get_loss(self, X, Yoh_):
        probs = self.forward(X)
        logprobs = torch.log(probs) * Yoh_
        logsum = torch.sum(logprobs, dim=1)
        logprobs_mean = torch.mean(logsum)

        return -logprobs_mean


def train(model, X, Yoh_, param_niter, param_delta=0.1, param_reg=10e-4, verbose=True):
    """Arguments:
       - X: model inputs [NxD], type: torch.Tensor
       - Yoh_: ground truth [NxC], type: torch.Tensor
       - param_niter: number of training iterations
       - param_delta: learning rate
    """

    optimizer = SGD(model.parameters(), lr=param_delta)

    for i in range(param_niter):
        loss = model.get_loss(X, Yoh_) + param_reg * torch.norm(model.W)
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

    return np.argmax(model.forward(torch.from_numpy(X)).detach().numpy(), axis=1)


if __name__ == "__main__":
    np.random.seed(100)

    X, Y_ = data.sample_gauss_2d(3, 100)
    x = torch.from_numpy(X)
    Yoh_ = torch.from_numpy(data.class_to_onehot(Y_))

    ptlr = PTLogreg(x.shape[1], Yoh_.shape[1])

    train(ptlr, x, Yoh_, 10000, 0.5)

    Y = eval(ptlr, X)

    accuracy, precision, confusion_matrix = data.eval_perf_multi(Y, Y_)
    print('Accuracy: ', accuracy)
    print('Precision: ', precision)
    print('Confusion matrix:\n', confusion_matrix)

    bbox = (np.min(X, axis=0), np.max(X, axis=0))
    decision = lambda X: eval(ptlr, X)
    print(eval(ptlr, X))
    data.graph_surface(decision, bbox, offset=0)

    data.graph_data(X, Y_, Y, special=[])
    plt.show()