import data
import torch
import torchvision
import matplotlib.pyplot as plt
import pt_deep


def get_data():
    dataset_root = 'data' 
    mnist_train = torchvision.datasets.MNIST(dataset_root, train=True, download=False)
    mnist_test = torchvision.datasets.MNIST(dataset_root, train=False, download=False)

    return mnist_train, mnist_test


def task_one():

    train_data, test_data = get_data()
    X = train_data.data / 255.0
    Yoh_ = torch.from_numpy(data.class_to_onehot(train_data.targets.detach().numpy()))

    model = pt_deep.PTDeep([784, 10], torch.relu)
    pt_deep.train(model, X.view(-1, 784), Yoh_, 3000)

    weights = model.weights[0].detach().numpy().T.reshape((-1, 28, 28))

    fig, ax = plt.subplots(2, 5)

    for i, digit_weight in enumerate(weights):
        ax[i // 5 - 1, i % 5].imshow(digit_weight)

    plt.show()


if __name__ == "__main__":
    task_one()