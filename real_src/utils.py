import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from time import time
from functools import wraps
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from geomloss import SamplesLoss


def data_distance(s_train, x_train):
    wloss = SamplesLoss("sinkhorn", p=1, blur=0.01)
    data0 = torch.FloatTensor(x_train[s_train[:, 0] == 0])[:5000]
    data1 = torch.FloatTensor(x_train[s_train[:, 0] == 1])[:5000]
    print(f"X0-X1-W-dis:", wloss(data0, data1))
    

def tensor(x):
    return torch.FloatTensor(x)


def count_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time()
        result = func(*args, **kwargs)
        end_time = time()
        duration = end_time - start_time
        print(f"Time: {duration:5.2f}s")
        return result
    return wrapper


def logistic(x):
    return torch.log(1 + torch.exp(x))


def compute_short_fairness(model, s_mb, x_mb):
    neg_x = x_mb[s_mb.squeeze() == 0]

    neg_s = torch.zeros((neg_x.shape[0], 1)).to(x_mb.device)
    y_neg = model(neg_s, neg_x)
    
    pos_s = torch.ones((neg_x.shape[0], 1)).to(x_mb.device)
    y_pos = model(pos_s, neg_x)
    
    p_y_neg = (y_neg >= 0.5).sum() / len(y_neg)
    p_y_pos = (y_pos >= 0.5).sum() / len(y_pos)

    fair1 = torch.mean(logistic(y_pos) + logistic(-y_neg) - 1)
    fair2 = torch.mean(logistic(y_neg) + logistic(-y_pos) - 1)

    return (y_pos - y_neg).mean().abs(), p_y_pos - p_y_neg

def compute_long_fairness(model, s_mb, x_mb):
    neg_x = x_mb[s_mb.squeeze() == 0]
    neg_s = torch.zeros((neg_x.shape[0], 1)).to(x_mb.device)
    pred_y = model.predict(neg_s, neg_x)
    p_y_neg = sum(pred_y == 1) / len(pred_y)

    pos_x = x_mb[s_mb.squeeze() == 1]
    neg_s = torch.zeros((pos_x.shape[0], 1)).to(x_mb.device)
    pred_y = model.predict(neg_s, pos_x)
    p_y_pos = sum(pred_y == 1) / len(pred_y)

    return p_y_pos - p_y_neg


def demographic_parity(sensi, pred_y):

    s0 = sum(sensi.squeeze() == 0)
    s1 = sum(sensi.squeeze() == 1)
    y0 = sum(pred_y.squeeze() == 0)
    y1 = sum(pred_y.squeeze() == 1)
    y1_s0 = sum(pred_y[sensi.squeeze() == 0].squeeze() == 1) / s0
    y1_s1 = sum(pred_y[sensi.squeeze() == 1].squeeze() == 1) / s1
    print(f"#(S=0): {s0}, #(S=1): {s1}, #(y0): {y0}, #(y1): {y1}, P(y=1|s=0)={y1_s0:.3f}, P(y=1|s=1)={y1_s1:.3f}")
    return y1_s1 - y1_s0


def visualize_step_data(data, label, step, sample_size=10000):
    np.random.seed(0)

    batch, seq_len, n_dim = data.shape
    idx = np.random.permutation(batch)[:sample_size]
    data = data[idx, step]
    label = label[idx, step].flatten()

    pca = PCA(n_components=2, random_state=0)
    pca_sample = pca.fit_transform(data)

    tsne = TSNE(n_components=2, perplexity=20, verbose=0, random_state=0)
    tsne_sample = tsne.fit_transform(data)

    plot_data(pca_sample, tsne_sample, label)


def visualize_step_data(sensi, data, label, step, sample_size=1000, path=None):
    rng = np.random.RandomState(0)
    shuffled = rng.permutation(len(sensi))[:sample_size]

    tsne = TSNE(n_components=2, verbose=1, learning_rate='auto', init='random', random_state=0)
    data = tsne.fit_transform(data[:, step][shuffled])
    
    x_draw = data
    y_draw = label[:, step][shuffled].squeeze()
    s_draw = sensi[shuffled].squeeze()

    X_s_0 = x_draw[s_draw == 0.0]
    X_s_1 = x_draw[s_draw == 1.0]
    y_s_0 = y_draw[s_draw == 0.0]
    y_s_1 = y_draw[s_draw == 1.0]

    MARKER_SIZE = 100
    cmap_points = "coolwarm_r"
    fig, ax = plt.subplots(figsize=(6, 4.7))

    ax.scatter(X_s_0[y_s_0 == 1.0][:, 0], X_s_0[y_s_0 == 1.0][:, 1], c=np.ones_like(y_s_0[y_s_0 == 1.0]), cmap=cmap_points, alpha=0.9, marker='+', s=MARKER_SIZE, linewidth=2.5, norm=MidpointNormalize(midpoint=0,vmax=1,vmin=-1), label='s=0, y=1')
    ax.scatter(X_s_0[y_s_0 == 0.0][:, 0], X_s_0[y_s_0 == 0.0][:, 1], c=np.ones_like(y_s_0[y_s_0 == 0.0]), cmap=cmap_points, alpha=0.9,marker='_', s=MARKER_SIZE, linewidth=2.5, norm=MidpointNormalize(midpoint=0,vmax=1, vmin=-1), label='s=0, y=0')
    ax.scatter(X_s_1[y_s_1 == 1.0][:, 0], X_s_1[y_s_1 == 1.0][:, 1], c=-1*np.ones_like(y_s_1[y_s_1 == 1.0]), cmap=cmap_points, alpha=0.9,marker='+', s=MARKER_SIZE, linewidth=2.5, norm=MidpointNormalize(midpoint=0,vmax=1,vmin=-1), label='s=1, y=1')
    ax.scatter(X_s_1[y_s_1 == 0.0][:, 0], X_s_1[y_s_1 == 0.0][:, 1], c=-1*np.ones_like(y_s_1[y_s_1 == 0.0]), cmap=cmap_points,alpha=0.9, marker='_', s=MARKER_SIZE, linewidth=2.5, norm=MidpointNormalize(midpoint=0,vmax=1,vmin=-1), label='s=1, y=0')

    plt.locator_params(nbins=5)
    plt.legend(loc=2, fontsize=16)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['bottom'].set_linewidth(1.2)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['top'].set_linewidth(1.2)
    ax.spines['right'].set_linewidth(1.2)
    ax.set_xlabel('')
    ax.set_ylabel('')
    plt.tight_layout()

    if path:
        fig.savefig(fname=path, dpi=300)

    plt.show()


# set the colormap and centre the colorbar
class MidpointNormalize(mcolors.Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        mcolors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))


def visualization(real_data, gen_data, path, sample_size=250):
    np.random.seed(0)
    batch, seq_len, n_dim = real_data.shape
    idx = np.random.permutation(batch)[:sample_size]

    real_sample = real_data[idx]
    syn_sample = gen_data[idx]
    print(f"Real shape: {real_sample.shape}")
    print(f"Syn shape: {syn_sample.shape}")
    real_sample_2d = real_sample.reshape(-1, n_dim)
    syn_sample_2d = syn_sample.reshape(-1, n_dim)

    tsne_data = np.concatenate((real_sample_2d, syn_sample_2d), axis=0)
    tsne = TSNE(n_components=2, verbose=1, perplexity=40)
    tsne_result = tsne.fit_transform(tsne_data)
    tsne_result = pd.DataFrame(tsne_result, columns=['X', 'Y']).assign(Data='Real')
    tsne_result.loc[sample_size*seq_len:, 'Data'] = 'Generated'
    
    fig, axes = plt.subplots(ncols=1, figsize=(6, 5))
    sns.scatterplot(x='X', y='Y', data=tsne_result, hue='Data', style='Data')
    sns.despine()
    axes.set_xticks([])
    axes.set_yticks([])
    axes.spines['left'].set_visible(True)
    axes.spines['right'].set_visible(True)
    axes.spines['bottom'].set_visible(True)
    axes.spines['top'].set_visible(True)
    axes.spines['bottom'].set_linewidth(1.2)
    axes.spines['left'].set_linewidth(1.2)
    axes.spines['top'].set_linewidth(1.2)
    axes.spines['right'].set_linewidth(1.2)
    axes.set_xlabel('')
    axes.set_ylabel('')
    axes.legend(fontsize=22, loc=2)
    
    axes.set_title('T-SNE of Taiwan Dataset', fontsize=22, pad=20)
    fig.tight_layout()
    fig.subplots_adjust(top=.88)
    fig.savefig(fname=path, dpi=300)
    plt.show()

def plot_data(pca_data, tsne_data, label):
    fig, axes = plt.subplots(ncols=2, figsize=(14, 5))
    sns.scatterplot(x=pca_data[:, 0], y=pca_data[:, 1], hue=label, ax=axes[0])
    sns.despine()
    axes[0].set_title('PCA Result')

    sns.scatterplot(x=tsne_data[:, 0], y=tsne_data[:, 1], hue=label, ax=axes[1])
    sns.despine()
    axes[1].set_title('t-SNE Result')

    fig.tight_layout()
    fig.subplots_adjust(top=.88)
    plt.show()
