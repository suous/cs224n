import itertools
import os
import random
from collections import Counter

import numpy as np
import pandas as pd


def read_cs224_sentences(path, unk_token="<unk>", **kwargs):
    df = pd.read_csv(
        path,
        sep="\t",
        usecols=["sentence"],
        converters={"sentence": lambda s: s.strip().lower().split()},
        **kwargs,
    )
    df["length"] = df["sentence"].apply(len)
    tokens = list(itertools.chain(*df.sentence.to_list(), [unk_token]))
    token_keys = dict.fromkeys(tokens).keys()
    token_dict = dict(zip(token_keys, range(len(token_keys))))
    token_freq = Counter(tokens)
    return df, tokens, token_dict, token_freq


def seed_everything(seed: int = 21):
    """
    Seed all random number generators for reproducibility.

    Parameters
    ----------
    seed : int, optional, default=21
        The seed to use.
    """

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def one_hot_encode(x):
    """
    One-hot encode a 1D array.

    This function takes a 1D array of integers and returns a 2D array of
    one-hot encoded values. The input array is assumed to contain integers
    in the range 0 to k-1, where k is the maximum value in the array.

    Parameters
    ----------
    x : ndarray
        The input array to encode.

    Returns
    -------
    ndarray
        The one-hot encoded array.

    Example
    -------
    >>> one_hot_encode(np.array([1, 0, 2, 3]))
    array([[0., 1., 0., 0.],
           [1., 0., 0., 0.],
           [0., 0., 1., 0.],
           [0., 0., 0., 1.]])
    """

    return np.eye(x.max() + 1)[x]


def moving_average(x, win=10):
    """
    Calculate the moving average of a 1D array.

    Parameters
    ----------
    x : ndarray
        The input array.
    win : int, optional, default=10
        The window size.

    Returns
    -------
    ndarray
        The moving average of the input array.
    """
    return np.convolve(x, np.ones(win), "valid") / win


def sigmoid(z):
    """
    Calculate the sigmoid function for a given input z.

    Parameters
    ----------
    z : np.ndarray
        The input to the sigmoid function.
    Returns
    -------
    ndarray
        The output of the sigmoid function applied element-wise to the input array.
    """

    return 1 / (1 + np.exp(-z))


def softmax(x, axis=None):
    """
    Calculate the softmax function for a given input x.

    Parameters
    ----------
    x : np.ndarray
        The input to the softmax function.
    axis : int or tuple of ints, optional
        Axis to compute values along. Default is None and softmax will be
        computed over the entire array `x`.
    Returns
    -------
    ndarray
        The output of the softmax function applied element-wise to the input array.
    """

    x_max = np.amax(x, axis=axis, keepdims=True)
    exp_x_shifted = np.exp(x - x_max)
    return exp_x_shifted / np.sum(exp_x_shifted, axis=axis, keepdims=True)


# fmt: off
def pca(x, n_components=2):
    """
    Principal component analysis.

    Parameters
    ----------
    x : np.ndarray
        The input array.
    n_components : int, optional, default=2
        The number of components to keep.

    Returns
    -------
    np.ndarray
    """
    x = x - x.mean(axis=0)                         # (m, n)
    cov = np.cov(x, rowvar=False)                  # (n, n)
    eigen_vals, eigen_vecs = np.linalg.eigh(cov)   # (n, 1), (n, n)
    idx_sorted = np.argsort(eigen_vals)[::-1]      # (n, 1)
    eigen_vecs_sorted = eigen_vecs[:, idx_sorted]  # (n, n)
    eigen_vecs_subset = eigen_vecs_sorted[:, :n_components]  # (n, n_components)
    return x @ eigen_vecs_subset
# fmt: on
