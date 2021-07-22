import numpy as np
import data
import matplotlib.pyplot as plt


def fcann2_train(X, Y_, param_niter=1e5, param_delta=0.05, param_lambda=1e-3, hidden_size=5, verbose=False):
    C = max(Y_) + 1
    D = len(X[0])
    N = len(X)

    W1 = np.random.randn(hidden_size, D)  # H x D
    b1 = np.zeros((1, hidden_size))       # 1 x H
    W2 = np.random.randn(C, hidden_size)  # C x H
    b2 = np.zeros((1, C))                 # 1 x C

    for i in range(int(param_niter)):
        scores1 = np.dot(X, W1.T) + b1                               # N x H
        # ReLU
        h1 = np.maximum(0, scores1)                                  # N x H
        scores2 = np.dot(h1, W2.T) + b2                             # N x C
        expscores = np.exp(scores2)                                  # N x C
        sumexp = np.sum(expscores, axis=1, keepdims=True)

        probs = expscores / sumexp                                   # N x C

        correct_probs = probs[range(N), Y_]
        logprobs = -np.log(correct_probs)

        # regulariziran gubitak
        loss = np.sum(logprobs) / N + \
            param_lambda * (np.sum(np.square(W1)) + np.sum(np.square(W2)))

        # dijagnostički ispis
        if verbose and i % 10 == 0:
            print("iteration {}: loss {}".format(i, loss))

        Gs2 = probs
        Gs2[range(N), Y_] -= 1                                      # N x C
        Gs2 /= N

        grad_W2 = np.dot(Gs2.T, h1)
        grad_b2 = np.sum(Gs2, axis=0, keepdims=True)                # C x 1

        Gs1 = np.dot(Gs2, W2)                                       # N x H
        Gs1[h1 <= 0] = 0

        grad_W1 = np.dot(Gs1.T, X)                                  # H x D
        grad_b1 = np.sum(Gs1, axis=0, keepdims=True)                  # H x 1

        # poboljšani parametri
        W1 += -param_delta * grad_W1
        b1 += -param_delta * grad_b1
        W2 += -param_delta * grad_W2
        b2 += -param_delta * grad_b2

    return W1, b1, W2, b2


def fcann2_classify(X, W1, b1, W2, b2):
    scores1 = np.dot(X, W1.T) + b1
    h1 = np.maximum(0, scores1)
    expscores = np.exp(np.dot(h1, W2.T) + b2)
    sumexp = np.sum(expscores, axis=1, keepdims=True)

    return (expscores / sumexp)


if __name__ == "__main__":
    np.random.seed(100)

    X, Y_ = data.sample_gmm_2d(6, 2, 10)
    W1, b1, W2, b2 = fcann2_train(X, Y_)
    print("b1: ", b1)
    probs = fcann2_classify(X, W1, b1, W2, b2)
    Y = np.argmax(probs, axis=1)
    # ispiši performansu (preciznost i odziv po razredima)
    accuracy, precision, confusion_matrix = data.eval_perf_multi(Y, Y_)
    print('Accuracy:', accuracy)
    print('Precision:', precision)
    print('Confusion matrix:\n', confusion_matrix)

    # iscrtaj rezultate, decizijsku plohu
    rect = (np.min(X, axis=0), np.max(X, axis=0))
    def decision(X): return fcann2_classify(X, W1, b1, W2, b2)[:, 0]
    data.graph_surface(decision, rect, offset=0.5)

    # graph the data points
    data.graph_data(X, Y_, Y, special=[])
    plt.show()
