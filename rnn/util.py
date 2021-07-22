import torch
import numpy as np

TRAIN_PATH = "data/sst_train_raw.csv"
VALID_PATH = "data/sst_valid_raw.csv"
TEST_PATH = "data/sst_test_raw.csv"
GLOVE_PATH = "data/sst_glove_6b_300d.txt"


def evaluate(model, data, loss_function):
    with torch.no_grad():
        confusion_matrix = np.zeros([2, 2])
        for batch in data:
            y_pred = model.infer(batch[0]).numpy().flatten()
            y_real = torch.tensor(list(batch[1])).numpy()
            batch_size = len(y_pred)
            for i in range(batch_size):
                confusion_matrix[y_real[i]][y_pred[i]] += 1

        accuracy = np.sum(np.diag(confusion_matrix)) / np.sum(confusion_matrix)
        recall = confusion_matrix[1][1] / (confusion_matrix[1][0] + confusion_matrix[1][1])
        precision = confusion_matrix[1][1] / (confusion_matrix[0][1] + confusion_matrix[1][1])
        f1 = 2 * precision * recall / (precision + recall)
        print("Accuracy: %.2f" % accuracy)
        print("f1: %.2f" % f1)
        print("Confusion matrix:\n", confusion_matrix)
