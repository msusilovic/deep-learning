from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.utils.data as data
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms

import util

SAVE_DIR = Path(__file__).parent / 'out'
DATA_DIR = Path(__file__).parent / 'data'

MAX_EPOCHS = 8
BATCH_SIZE = 50


class ConvolutionalModel():
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 5, padding=2)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5, padding=2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc3 = nn.Linear(1568, 512)
        self.logits = nn.Linear(512, 10)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear) and m is not self.fc_logits:
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
        self.logits.reset_parameters()

    def forward(self, x):
        h = self.conv1(x)  # Nx16x28x28
        h = self.pool1(h)  # Nx16x14x14
        h = torch.relu(h)  # Nx16x14x14
        h = self.conv2(h)  # Nx32x14x14
        h = self.pool2(h)  # Nx32x7x7
        h = torch.relu(h)  # Nx32x7x7
        h = h.view(h.shape[0], -1)  # Nx1568
        h = self.fc3(h)  # Nx512
        h = torch.relu(h)  # Nx512
        h = self.logits(h)  # Nx10

        return h

    def train(self, loss_function, num_examples, train_dataloader, val_dataloader, lr=0.1, weight_decay=0.01, verbose=False):
        optimizer = SGD([
            {"params": [*self.conv1.parameters(),
                        *self.conv2.parameters(),
                        *self.fc3.parameters()], "weight_decay": weight_decay},
            {"params": self.logits.parameters(), "weight_decay": 0}
        ], lr=lr)
        scheduler = StepLR(optimizer=optimizer, step_size=2, gamma=0.1)

        losses = []
        num_batches = num_examples / BATCH_SIZE

        for epoch in range(1, MAX_EPOCHS + 1):
            avg_loss = 0
            cnt_correct = 0
            for i, (x, y) in enumerate(train_dataloader):
                pred = self.forward(x)
                loss = loss_function(pred, y)
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
            scheduler.step()
            util.draw_conv_filters(epoch, i * BATCH_SIZE, self.conv1.weight, SAVE_DIR)
            print("Train accuracy = %.2f" % (cnt_correct / num_examples * 100))
            # evaluate validation set
            self.evaluate("Validation", val_dataloader, loss_function)

            losses.append(avg_loss / num_batches)

        plt.title("Kretanje gubitka kroz epohe")
        plt.plot(range(1, MAX_EPOCHS + 1), losses)
        plt.show()

    def evaluate(self, name, dataloader, loss_function):
        num_ex = len(dataloader.dataset)
        num_batches = num_ex / BATCH_SIZE
        avg_loss = 0
        cnt_correct = 0
        for i, (x, y) in enumerate(dataloader):
            pred = self.forward(x)
            loss = loss_function(pred, y)
            avg_loss += loss.detach().numpy()
            yp = torch.argmax(pred, dim=1)
            cnt_correct += (yp == y).sum().numpy()
        valid_acc = cnt_correct / num_ex * 100
        avg_loss /= num_batches
        print(name + " accuracy = %.2f" % valid_acc)
        print(name + " avg loss = %.2f\n" % avg_loss)


ds_train, ds_test = datasets.MNIST(DATA_DIR, train=True, download=False, transform=transforms.ToTensor()), datasets.MNIST(DATA_DIR, train=False, transform=transforms.ToTensor())
train, validate = data.random_split(ds_train, [55000, 5000])
train_dataloader = data.DataLoader(train, batch_size=BATCH_SIZE)
val_dataloader = data.DataLoader(validate, batch_size=BATCH_SIZE)


loss_function = nn.CrossEntropyLoss()
model = ConvolutionalModel()
num_examples = len(train_dataloader.dataset)
print(num_examples)
model.train(loss_function, num_examples, train_dataloader, val_dataloader, verbose=True)
