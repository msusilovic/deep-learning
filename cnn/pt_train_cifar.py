import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as data
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import ExponentialLR
from torchvision import datasets, transforms

import util

BATCH_SIZE = 50
DATA_DIR = 'cifar/'
SAVE_DIR = 'cout/'
MAX_EPOCHS = 10


class ConvolutionalModel:
    def __init__(self, input_depth):
        self.conv1 = nn.Conv2d(input_depth, 16, 5, padding=2)
        self.pool1 = nn.MaxPool2d(3, 2, padding=(1, 1))
        self.conv2 = nn.Conv2d(16, 32, 5, padding=2)
        self.pool2 = nn.MaxPool2d(3, 2, padding=(1, 1))
        self.fc1 = nn.Linear(2048, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        h = torch.relu(self.conv1(x))  # Nx16x32x32
        h = self.pool1(h)  # Nx16x16x16
        h = torch.relu(self.conv2(h))  # Nx32x16x16
        h = self.pool2(h)  # Nx32x8x8
        h = h.view(h.shape[0], -1)  # Nx2048
        h = torch.relu(self.fc1(h))
        h = torch.relu(self.fc2(h))
        h = self.fc3(h)

        return h

    def train(self, loss_function, train_loader, val_dataloader, lr=0.1, weight_decay=0.01, verbose=False):
        optimizer = SGD([
            {"params": [*self.conv1.parameters(),
                        *self.conv2.parameters(),
                        *self.fc1.parameters(),
                        *self.fc2.parameters()], "weight_decay": weight_decay},
            {"params": self.fc3.parameters(), "weight_decay": 0}
        ], lr=lr)
        scheduler = ExponentialLR(optimizer, gamma=0.8)

        num_examples = len(train_loader.dataset)
        train_loss = []
        valid_loss = []
        train_acc = []
        valid_acc = []
        l_rates = []
        num_batches = num_examples / BATCH_SIZE
        for epoch in range(1, MAX_EPOCHS + 1):
            avg_loss = 0
            cnt_correct = 0
            for i, (x, y) in enumerate(train_loader):
                pred = self.forward(x)
                loss = loss_function(pred, y.long())
                loss.backward()
                avg_loss += loss.detach().numpy()
                optimizer.step()
                optimizer.zero_grad()
                # compute classification accuracy
                yp = torch.argmax(pred, dim=1)
                cnt_correct += (yp == y).sum().numpy()
                if verbose and i % 5 == 0:
                    print("epoch %d, step %d/%d, batch loss = %.2f" % (epoch, i * BATCH_SIZE, num_examples, loss))
                if i > 0 and i % 50 == 0:
                    print("Train accuracy = %.2f" % (cnt_correct / ((i + 1) * BATCH_SIZE) * 100))
            l_rates.append(scheduler.get_lr())
            scheduler.step()
            util.draw_conv_filters(epoch, i * BATCH_SIZE, self.conv1.weight, SAVE_DIR)
            tr_accuracy = cnt_correct / num_examples
            print("Train accuracy = %.2f" % (tr_accuracy * 100))
            v_acc, v_loss = self.evaluate("Validation", val_dataloader, loss_function)
            train_loss.append(avg_loss / num_batches)
            valid_loss.append(v_loss)
            train_acc.append(tr_accuracy)
            valid_acc.append(v_acc)

        util.plot_loss(train_loss, valid_loss, train_acc, valid_acc, l_rates, SAVE_DIR)

    def evaluate(self, name, dataloader, loss_function):
        with torch.no_grad():
            for x, y in dataloader:
                num_examples = len(x)
                pred = self.forward(x)
                yp = torch.argmax(pred, dim=1).detach().numpy()
                ys = y.detach().numpy()
                num_classes = pred.shape[1]
                confusion_matrix = np.zeros([num_classes, num_classes])
                for i in range(num_examples):
                    confusion_matrix[ys[i]][yp[i]] += 1
                accuracy = np.sum(np.diag(confusion_matrix)) / num_examples
                loss = loss_function(pred, y)
                for i in range(num_classes):
                    print("Class " + str(i))
                    print("Precision: %.2f" % (confusion_matrix[i][i] / np.sum(confusion_matrix[:, i])) + ", Recall: %.2f" % (confusion_matrix[i][i] / np.sum(confusion_matrix[i, :])))
                print(name + " accuracy: %.2f" % accuracy)
                print(name + "confusion matrix:")
                print(confusion_matrix)
            return accuracy, loss


classes = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}

ds_train, ds_test = datasets.CIFAR10(DATA_DIR, train=True, download=False, transform=transforms.ToTensor()), datasets.CIFAR10(DATA_DIR, train=False, transform=transforms.ToTensor())
train, valid = data.random_split(ds_train, [45000, 5000])

x_valid = torch.FloatTensor(np.array(valid.dataset.data).transpose(0, 3, 1, 2))
y_valid = torch.FloatTensor(valid.dataset.targets)

train_dataloader = data.DataLoader(train, batch_size=BATCH_SIZE)
val_dataloader = data.DataLoader(valid, batch_size=5000)
test_dataloader = data.DataLoader(ds_test, batch_size=1)

model = ConvolutionalModel(3)
loss_function = nn.CrossEntropyLoss()
model.train(loss_function, train_dataloader, val_dataloader, verbose=True)

losses = []
for i, (x, y) in enumerate(test_dataloader):
    pred = model.forward(x)
    loss = loss_function(pred, y).item()
    ys = torch.argmax(pred)
    if y[0] != ys:
        losses.append((loss, pred.detach().numpy()[0], i))
losses = sorted(losses, key=lambda l: -l[0])[0:20]

f, axs = plt.subplots(5, 4)
f.tight_layout()
plt.subplots_adjust(top=0.75)

for ax, (loss, pred, i) in zip(axs.ravel(), losses):
    img = ds_test.data[i]
    y = ds_test.targets[i]
    top_pred = np.argpartition(pred, -3)[-3:]
    img = img.astype(np.uint8)
    ax.set_title(f'correct={classes[y]}\n predicted={classes[top_pred[0]]},{classes[top_pred[1]]},{classes[top_pred[2]]}', size=6)
    ax.imshow(img)
    ax.axis('off')
plt.show()
