import torch
import torch.nn as nn
from torch.optim import Adam

import copy
import numpy as np
from pathlib import Path
from sklearn.datasets import make_classification
from sklearn.cluster import KMeans

from syn_src.utils import count_time


def to_tensor(s, x, y=None):
    if torch.is_tensor(x):
        sx = torch.cat([s, x], dim=1)
    else:
        sx = np.concatenate([s, x], axis=1)
        sx = torch.FloatTensor(sx)
    if isinstance(y, np.ndarray):
        y = torch.FloatTensor(y)
        return sx, y
    return sx


class TrueModel(nn.Module):

    def __init__(self, hiddens, path, seed=0):
        super().__init__()
        self.path = path
        layers = []
        for in_dim, out_dim in zip(hiddens[:-1], hiddens[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU(inplace=True))
        layers.pop()
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)

        self.loss_fn = nn.BCELoss()
        self.optim = Adam(self.parameters())

    def forward(self, sx):
        return self.model(sx)

    def predict(self, s, x):
        sx = to_tensor(s, x)
        pred = self(sx)
        pred_y = pred.detach().round().cpu().numpy()
        return pred_y

    def fit(self, s, x, y, patience=10):
        sx, y = to_tensor(s, x, y)

        epoch, counter = 0, 0
        best_loss = float('inf')
        while True:
            pred = self(sx)
            loss = self.loss_fn(pred, y)

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            
            epoch += 1
            if loss.item() <= best_loss:
                torch.save(self.state_dict(), self.path)
                best_loss = loss.item()
                counter = 0
            else:
                counter += 1
                if counter == patience:
                    break
        print(f"TrueModel Fit Done in {epoch} epochs!")

    def sample(self, s, x, scale=0.8):
        sx = to_tensor(s, x)
        prob = self(sx)
        y = torch.bernoulli(prob * scale)
        return y.detach().cpu().numpy()


def generate_init_dataset(no, dim):
    X, y = make_classification(
        n_samples=no, n_features=dim, n_informative=4, n_redundant=1,
        n_clusters_per_class=2, n_classes=2, flip_y=0.01, class_sep=1.0,
        random_state=1
    )

    kmeans = KMeans(n_clusters=4, random_state=0).fit(X)
    s = [[1] if cluster in [0, 2] else [0] for cluster in kmeans.labels_]

    s = np.array(s)
    y = y.reshape(-1, 1)
    return s, X, y


def get_components_of_change(model, s, x):
    sx = to_tensor(s, x)
    sx.requires_grad = True
    
    prob = model(sx)
    loss = nn.BCELoss()(prob, torch.ones_like(prob))
    loss.backward()
    x_grad = (sx.grad[:, 1:]).detach().cpu().numpy()

    def process(grad):
        if max(abs(grad)) != 0.0:
            while max(abs(grad)) < 1.:
                grad *= 10
        return grad
    x_grad = np.array(list(map(process, x_grad)))

    sign = 2 * torch.bernoulli(prob) - 1
    return x_grad, sign.detach().cpu().numpy()


def generate_next_dataset(s0, x0, y0, model, seq_len, epsilon):
    max_grad = 2
    max_val = 20

    x, y = [x0], [y0]
    for i in range(seq_len - 1):
        nx = copy.deepcopy(x[-1])
        ny = copy.deepcopy(y[-1]).flatten()

        x_grad, sign = get_components_of_change(model, s0, nx)

        idx = np.squeeze(ny == 1) & np.squeeze(s0 == 1)
        nx[idx] -= np.clip(epsilon * sign[idx] * x_grad[idx], a_min=-max_grad, a_max=max_grad)
        idx = np.squeeze(ny == 1) & np.squeeze(s0 == 0)
        nx[idx] -= np.clip(epsilon * sign[idx] * x_grad[idx], a_min=-max_grad, a_max=max_grad)

        idx = np.squeeze(ny == 0) & np.squeeze(s0 == 1)
        nx[idx] -= np.clip(epsilon * sign[idx] * x_grad[idx], a_min=-max_grad, a_max=max_grad)
        idx = np.squeeze(ny == 0) & np.squeeze(s0 == 0)
        nx[idx] -= np.clip(2 * epsilon * sign[idx] * x_grad[idx], a_min=-max_grad, a_max=max_grad)

        nx = np.clip(nx, a_min=-max_val, a_max=max_val)

        pred_ny = model.sample(s0, nx)
        x.append(nx)
        y.append(pred_ny)

    return x, y


@count_time
def generate_sequential_datasets(no, dim, seq_len, hiddens, epsilon, device, path, seed=0):

    model = TrueModel(hiddens, path, seed)

    s0, x0, y0 = generate_init_dataset(no, dim)
    if not Path(path).exists():
        model.fit(s0, x0, y0)

    model.load_state_dict(torch.load(path, map_location=device))
    y0 = model.sample(s0, x0)

    x, y = generate_next_dataset(s0, x0, y0, model, seq_len, epsilon)

    x = np.transpose(np.array(x, dtype=np.float32), axes=(1, 0, 2))
    y = np.transpose(np.array(y, dtype=np.int32), axes=(1, 0, 2))
    return s0, x, y, model