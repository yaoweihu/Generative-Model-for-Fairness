import copy
import numpy as np
from sklearn import metrics
from geomloss import SamplesLoss

import torch
import torch.nn as nn
from torch.optim import Adam, RMSprop
from torch.utils.data import DataLoader, TensorDataset
from real_src.utils import *


wloss = SamplesLoss("sinkhorn", p=1, blur=0.01)


def generate_from_gan(batch_size, seq_len, loader, clf, G, device, idx=0):
    gen_s, gen_x, gen_y = [], [], []

    for s_mb, x_mb, y_mb in loader:
        x_mb = x_mb.to(device)
        z_mb = torch.randn(x_mb.shape[0], seq_len-1, x_mb.shape[-1]).to(device)

        gen_x_mb, gen_y_mb, _ = G(x_mb[:, idx], z_mb, s_mb, clf)
        
        gen_s.append(s_mb)
        gen_x.append(gen_x_mb)
        gen_y.append(gen_y_mb)

    return [gen_s, gen_x, gen_y]


@count_time
def risk_minimization(batch_size, seq_len, true_model, clf, G, loader, w_lambda, f_lambda, device, save_path):

    loss_fn = torch.nn.BCELoss()
    optim_C = torch.optim.Adam(clf.parameters(), lr=0.0005)

    num_epoch = 0
    patience = 20
    best_loss = float('inf')
    while num_epoch < 500:
        data_loader = generate_from_gan(batch_size, seq_len, loader, clf, G, device)

        loss, num = 0., 0
        old_param = clf.get_params()
        for s_mb, x_mb, y_mb in zip(*data_loader):
            s_mb = s_mb.to(device)

            loss_C, loss_F = 0., 0.
            for i in range(x_mb.size(1)):
                y_pred = y_mb[:, i]
                y_true = true_model.sample(s_mb, x_mb[:, i])
                y_true = torch.FloatTensor(y_true).to(device)

                loss_C += loss_fn(y_pred, y_true)
                loss_F += compute_short_fairness(clf, s_mb.detach(), x_mb[:, i].detach())[0]

            wwloss = wloss(x_mb[:,-1][s_mb.squeeze()==0], x_mb[:,-1][s_mb.squeeze()==1])
            loss_epoch = loss_C + f_lambda * loss_F + w_lambda * wwloss

            loss += loss_epoch
            num += x_mb.shape[0]

        optim_C.zero_grad()
        loss.backward()
        optim_C.step()

        num_epoch += 1
        new_param = clf.get_params()
        gap = np.linalg.norm(new_param - old_param)

        if loss <= best_loss:
            torch.save(clf.state_dict(), save_path)
            # print("Save")
            best_loss = loss
            counter = 0
        # print(f"epochs: {num_epoch}, loss: {loss:.5f}, gap: {gap:.5f}")


def valid_classifier(seq_len, true_model, clf, G, loader, device, idx=0, verbose=True):
    batchs = [len(s_mb) for s_mb, _, _ in loader]
    data_loader = generate_from_gan(batchs[0], seq_len, loader, clf, G, device, idx=idx)

    for s_mb, x_mb, y_mb in zip(*data_loader):
        s_mb = s_mb.to(device)
        x_mb = x_mb.to(device)

        for i in range(x_mb.size(1)):
            y_pred = y_mb[:, i].round().detach().cpu().numpy()
            y_true = true_model.sample(s_mb, x_mb[:, i])
            
            acc = metrics.accuracy_score(y_true, y_pred) * 100
            short_fair = compute_short_fairness(clf, s_mb, x_mb[:, i])[1]
            long_fair = compute_long_fairness(clf, s_mb, x_mb[:, i])
            w_loss = wloss(x_mb[:, i][s_mb.squeeze()==0], x_mb[:, i][s_mb.squeeze()==1])
        
            if verbose:
                print(f"Step:{i:6.0f}, ACC:{acc:6.3f}%, Short-Fair:{short_fair.item():6.3f}, Long-Fair:{long_fair.item():6.3f}, W-dist:{w_loss:6.4f}")
    return short_fair.item(), long_fair.item()