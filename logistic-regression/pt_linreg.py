import numpy as np
import torch
import torch.optim as optim

def linear_regression(x, y_, param_niter=100):
    a = torch.randn(1, requires_grad=True)
    b = torch.randn(1, requires_grad=True)

    X = torch.tensor(x)
    Y = torch.tensor(y_)

    optimizer = optim.SGD([a, b], lr=0.1)

    for i in range(100):
        Y_ = a*X + b

        diff = (Y-Y_)

        loss = torch.mean(diff**2)

        loss.backward()

        grad_a = - 2 * torch.mean((diff) * X)
        grad_b = - 2 * torch.mean(diff)

        optimizer.step()

        print(f'step: {i}')
        print(f'a_grad: {a.grad.detach().numpy()[0]:.4f}, b_grad: {b.grad.detach().numpy()[0]:.4f}')
        print(f'my_a_grad: {grad_a:.4f}, my_b_grad: {grad_b:.4f}')

        optimizer.zero_grad()

        print(f'loss:{loss}, Y_:{Y_}, a:{a}, b {b}\n')


if __name__ == "__main__":
    x = np.array([0, 1, 2, 3])
    y = np.array([1, 3, 5, 7])
    linear_regression(x, y)