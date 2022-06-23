import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold, datasets
import datetime

colors=['#F9F871','#FFC75F','#FFC75F','#FF6F91','#D65DB1','#845EC2','#008E9B','#2C73D2','#B0A8B9','#4B4453','#4FFBDF','#FFD0FF','#008AC4']

def get_datetime_str(style='dt'):
    cur_time = datetime.datetime.now()

    date_str = cur_time.strftime('%y_%m_%d_')
    time_str = cur_time.strftime('%H_%M_%S')

    if style == 'data':
        return date_str
    elif style == 'time':
        return time_str
    else:
        return date_str + time_str


def T_SNE(X, y, if_save_image=False, **kwargs):
    """
    This function is used to draw T-SNE visualization, supports np.ndarray and torch.Tensor types

    X: shape(number,vector's length), input samples
    y: shape(number,1) or shape(number,), labels
    if_save_image: whether to save the picture
    title: the title of T-SNE picture
    n_components: 2 or 3, where 2 refers to 2d, 3 refers to 3d, respectively
    init: pca or random
    perplexity: consider the number of adjacent points during optimization
    cmap: True or False
    """
    if isinstance(X, torch.Tensor):
        X = X.clone().detach().cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.clone().detach().cpu().numpy()
    assert isinstance(X, np.ndarray) and isinstance(y, np.ndarray), "X and y should be numpy ndarray type!"
    y = y.astype(int)
    if y.ndim == 2:
        y = y.reshape(-1)
    tsne = manifold.TSNE(n_components=kwargs.get("n_components", 2),
                         init=kwargs.get("init", 'pca'),
                         random_state=kwargs.get("random_state", 501),perplexity=kwargs.get("perplexity",30))
    X_tsne = tsne.fit_transform(X)
    print("Org data dimension is {}.Embedded data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)
    fig = plt.figure(figsize=(10, 10))
    if kwargs.get("n_components", 2) == 2:
        ax = fig.add_subplot(111)
        if kwargs.get("cmap", False):
            ax.scatter(X_norm[:, 0], X_norm[:, 1], cmap='viridis',alpha=0.9)
        else:
            ax.scatter(X_norm[:, 0], X_norm[:, 1], s=kwargs.get("s", 5), c=[colors[i] for i in y], cmap='viridis',alpha=0.9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("X")
        ax.set_ylabel("Y ", rotation=0)
        ax.set_title(kwargs.get("title", ""))
    else:
        ax = fig.add_subplot(111, projection='3d')
        if kwargs.get("cmap", False):
            ax.scatter(X[:, 0], X[:, 1], X[:, 2],s=kwargs.get("s", 5), cmap='viridis',alpha=0.9)
        else:
            ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=[colors[i] for i in y], cmap='viridis',alpha=0.9)
        if kwargs.get("view_init", None) != None:
            ax.view_init(kwargs["view_init"][0], kwargs["view_init"][1])
        else:
            ax.view_init(4, -72)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_xlabel("X")
        ax.set_ylabel("Y ", rotation=0)
        ax.set_zlabel("Z ", rotation=0)

        ax.set_title(kwargs.get("title", ""))
    if if_save_image:
        plt.savefig(get_datetime_str() + ".png")
    plt.show()

