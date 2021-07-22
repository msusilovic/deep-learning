import data
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt


class KSVMWrap:
    def __init__(self, X, Y_, param_svm_c=1, param_svm_gamma='auto'):
        self.svm = svm.SVC(C=param_svm_c, gamma=param_svm_gamma)
        self.svm.fit(X, Y_)

    def predict(self, X):
        return self.svm.predict(X)

    def get_scores(self, X):
        return self.svm.decision_function(X)

    def support(self):
        return self.svm.support_


if __name__ == "__main__":
    np.random.seed(100)

    X, Y_ = data.sample_gmm_2d(6, 2, 10)
    ksvm_wrap = KSVMWrap(X, Y_)

    Y = ksvm_wrap.predict(X)
    accuracy, precision, confusion_matrix = data.eval_perf_multi(Y, Y_)
    print('Accuracy: ', accuracy)
    print('Precision: ', precision)
    print('Confusion matrix:\n', confusion_matrix)

    # iscrtaj rezultate, decizijsku plohu
    bbox = (np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(lambda X: ksvm_wrap.predict(X), bbox, offset=0.5)

    # graph the data points
    data.graph_data(X, Y_, Y, special=[ksvm_wrap.support()])
    plt.show()

