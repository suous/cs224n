import math
import random
from collections import defaultdict
from functools import partial

import numpy as np
from utils import read_cs224_sentences, sigmoid, softmax


class DataLoader:
    def __init__(self, path, threshold=1e-5, table_size=1000000, unk="UNK"):
        self.path = path
        self._table_size = table_size
        self._unk = unk
        self.df, self._tokens, self._token_dict, self._token_freq = read_cs224_sentences(path, unk_token=unk)
        self._reject_prob = np.asarray(
            [max(0, 1 - np.sqrt(threshold * self.num_words / self.token_freq[w])) for w in self.token_list]
        )

        self._sentences = self.df.sentence.to_list()
        all_sentences = [
            [w for w in s if random.random() >= self._reject_prob[self.token_dict[w]]] for s in self.sentences * 30
        ]
        self._all_sentences = [s for s in all_sentences if len(s) > 1]
        self._sample_table = None
        self._context = None
        self._datasets = None

    def sentence_to_index(self, sentence):
        return [self.token_dict.get(w, self.token_dict[self._unk]) for w in sentence]

    def get_random_context(self, num_context=5, batch_size=50):
        if not hasattr(self, "_context") or self._context is None or self._context.shape[1] != (2 * num_context + 1):
            sentences = self.df.sentence.apply(
                lambda s: [self._unk] * num_context + s + [self._unk] * num_context
            ).to_list()
            self._context = np.concatenate(
                [
                    np.lib.stride_tricks.sliding_window_view(self.sentence_to_index(s), num_context * 2 + 1)
                    for s in sentences
                ]
            )
            center_word = self._context[:, num_context]
            context = np.concatenate([self._context[:, :num_context], self._context[:, num_context + 1 :]], axis=1)
            self._datasets = (center_word, context)

        center_word, context = self._datasets
        random_indices = np.random.randint(0, len(center_word), batch_size)
        return center_word[random_indices], context[random_indices]

    def get_negative_samples(self, outside_word_indices, k=10):
        batch_size, _ = outside_word_indices.shape
        negative_sample_word_index = np.zeros((batch_size, k), dtype=np.int32)
        for b in range(batch_size):
            for i in range(k):
                while idx := self.sample_token_index:
                    if idx not in outside_word_indices[b]:
                        break
                negative_sample_word_index[b, i] = idx
        return negative_sample_word_index

    @property
    def sample_table(self):
        if hasattr(self, "_sample_table") and self._sample_table is not None:
            return self._sample_table

        sampling_freq = np.asarray([self.token_freq.get(w, 0) ** 0.75 for w in self.token_list])
        sampling_freq /= np.sum(sampling_freq)
        sampling_freq = np.cumsum(sampling_freq) * self._table_size

        self._sample_table = [0] * self._table_size

        j = 0
        for i in range(self._table_size):
            while i > sampling_freq[j]:
                j += 1
            self._sample_table[i] = j

        return self._sample_table

    @property
    def tokens(self):
        return self._tokens

    @property
    def token_dict(self):
        return self._token_dict

    @property
    def token_freq(self):
        return self._token_freq

    @property
    def token_list(self):
        return list(self._token_dict.keys())

    @property
    def num_words(self):
        return len(self.tokens)

    @property
    def num_unique_words(self):
        return len(self.token_list)

    @property
    def sentences(self):
        return self._sentences

    @property
    def all_sentences(self):
        return self._all_sentences

    @property
    def sample_token_index(self):
        return self.sample_table[random.randint(0, self._table_size - 1)]


class SkipGram:
    def __init__(self, dataset, vec_dim=100, num_context=5, method="negative_sampling", k=10):
        assert hasattr(dataset, "get_random_context")
        assert hasattr(dataset, "get_negative_samples")
        assert hasattr(dataset, "token_dict")
        assert hasattr(dataset, "num_unique_words")
        assert method in ["negative_sampling", "naive_softmax"]

        self.word2index = dataset.token_dict
        self.dataset = dataset
        self.vec_dim = vec_dim
        self.num_context = num_context

        std = np.sqrt(2.0 / (dataset.num_unique_words + vec_dim))
        init_bound = np.sqrt(3.0) * std
        self.center_word_vectors = np.random.uniform(-init_bound, init_bound, (dataset.num_unique_words, vec_dim))
        self.outside_word_vectors = np.random.uniform(-init_bound, init_bound, (dataset.num_unique_words, vec_dim))

        self._forward = (
            partial(self.negative_sampling_loss_and_gradient, k=k)
            if method == "negative_sampling"
            else self.naive_softmax_loss_and_gradient
        )

    # ruff: noqa: PLR0913
    def fit(self, epochs=100, batch_size=32, lr=1e-4, verbose=True, print_every=10, anneal_every=1000, save_every=1000):
        """
        Train the model.

        Parameters
        ----------
        epochs : int, default=100
            The number of epochs to train.
        batch_size : int, default=32
            The batch size.
        lr : float, default=1e-4
            The learning rate.
        verbose : bool, default=True
            Whether to print the training progress.
        print_every : int, default=10
            Print the training progress every `print_every` epochs.
        anneal_every : int, default=1000
            Anneal the learning rate every `anneal_every` epochs.
        save_every : int, default=1000
            Save the model every `save_every` epochs.

        Returns
        -------
        losses : list of float
            The training losses for each epoch.
        """
        num_digits = int(math.log10(epochs)) + 1
        losses = defaultdict(list)
        moving_loss = None
        for epoch in range(epochs):
            loss, grad_center_word, grad_outside_word = self.train_one_epoch(batch_size=batch_size)
            self.center_word_vectors -= lr * grad_center_word
            self.outside_word_vectors -= lr * grad_outside_word

            if moving_loss is None:
                moving_loss = loss
            else:
                moving_loss = 0.95 * moving_loss + 0.05 * loss
            losses["loss"].append(loss)
            losses["moving_loss"].append(moving_loss)

            if verbose and (epoch + 1) % print_every == 0:
                print(f"Epoch {epoch+1:0{num_digits}d}/{epochs} Loss: {loss:.4f} Moving Loss: {moving_loss:.4f}")

            if (epoch + 1) % anneal_every == 0:
                lr /= 2

            if (epoch + 1) % save_every == 0:
                np.save(f"saved_center_word_vectors_{epoch+1}.npy", self.center_word_vectors)
                np.save(f"saved_outside_word_vectors_{epoch+1}.npy", self.outside_word_vectors)

        return losses

    def train_one_epoch(self, batch_size):
        center_word_indices, outside_word_indices = self.dataset.get_random_context(
            num_context=self.num_context, batch_size=batch_size
        )
        # center_word_indices: (batch_size,)
        # outside_word_indices: (batch_size, 2*num_context)
        loss, grad_center_words, grad_outside_words = self._forward(
            center_word_vectors=self.center_word_vectors[center_word_indices],
            outside_word_indices=outside_word_indices,
        )
        grad_center_word_vectors = np.zeros_like(self.center_word_vectors)
        grad_center_word_vectors[center_word_indices] = grad_center_words
        return loss.mean(), grad_center_word_vectors, grad_outside_words

    def naive_softmax_loss_and_gradient(
        self,
        center_word_vectors,
        outside_word_indices,
    ):
        """
        Naive Softmax loss & gradient function for word2vec models.

        Parameters
        ----------
        center_word_vectors : np.ndarray, shape=(batch_size,vec_dim)
            The center word vectors.
        outside_word_indices : np.ndarray, shape=(batch_size,num_outside_words)
            The index of the outside word.

        Returns
        -------
        loss : float
            The softmax loss.
        grad_center_vec : np.ndarray, shape=(batch_size,vec_dim)
            The gradient of the center word vector.
        grad_outside_vec : np.ndarray, shape=(num_unique_words,vec_dim)
            The gradient of the outside word vectors.
        """

        # v_c ->  center_word_vectors:    (batch_size,num_outside_words,vec_dim)
        # o   ->  outside_word_indices:   (batch_size,num_outside_words)
        # U   ->  outside_word_vectors:   (num_unique_words,vec_dim)
        # y   ->  target:                 (batch_size, num_outside_words, num_unique_words)
        # z   ->  np.dot(v_c, U.T):       (batch_size, num_outside_words, num_unique_words)

        center_word_vectors = np.repeat(center_word_vectors[:, None, :], outside_word_indices.shape[1], axis=1)
        # (batch_size, num_outside_words, vec_dim)
        z = center_word_vectors @ self.outside_word_vectors.T  # (batch_size, num_outside_words, num_unique_words)
        u_o = self.outside_word_vectors[outside_word_indices]  # (batch_size, num_outside_words, vec_dim)
        y_hat = softmax(z, axis=2)  # (batch_size, num_outside_words, num_unique_words)

        # fmt: off
        # ruff: noqa: E501
        loss = (np.log(np.exp(z).sum(axis=2)) - np.einsum("kon,kon->ko", center_word_vectors, u_o, optimize="greedy")).sum(axis=1)  # (batch_size,)
        # fmt: on

        y = np.zeros_like(y_hat)  # (batch_size, num_outside_words, num_unique_words)
        y[np.arange(y.shape[0])[:, None], np.arange(y.shape[1]), outside_word_indices] = 1

        grad_center_vec = ((y_hat - y) @ self.outside_word_vectors).sum(axis=1)
        grad_outside_vec = np.einsum("kom,kon->mn", y_hat - y, center_word_vectors, optimize="greedy")

        return loss, grad_center_vec, grad_outside_vec

    def negative_sampling_loss_and_gradient(self, center_word_vectors, outside_word_indices, k=10):
        """
        Negative sampling loss & gradient function for word2vec models.

        Parameters
        ----------
        center_word_vectors : np.ndarray, shape=(batch_size,vec_dim)
            The center word vectors.
        outside_word_indices : np.ndarray, shape=(batch_size,num_outside_words)
            The index of the outside word.
        k : int, optional (default=10)
            The number of negative samples to take.

        Returns
        -------
        loss : float
            The softmax loss.
        grad_center_vec : np.ndarray, shape=(batch_size,vec_dim)
            The gradient of the center word vector.
        grad_outside_vec : np.ndarray, shape=(num_unique_words,vec_dim)
            The gradient of the outside word vectors.
        """

        _, num_outside_words = outside_word_indices.shape
        negative_sample_indices = self.dataset.get_negative_samples(
            outside_word_indices, k=k * num_outside_words
        )  # (batch_size, k*num_outside_words)
        indices = np.hstack((outside_word_indices, negative_sample_indices))  # (batch_size, (k+1)*num_outside_words)

        u_o = self.outside_word_vectors[outside_word_indices]  # (batch_size, num_outside_words, vec_dim)
        u_n = self.outside_word_vectors[negative_sample_indices]  # (batch_size, k*num_outside_words, vec_dim)

        u = np.concatenate((u_o, -u_n), axis=1)  # (batch_size, (k+1)*num_outside_words, vec_dim)

        center_word_vectors = np.repeat(center_word_vectors[:, None, :], outside_word_indices.shape[1], axis=1)
        # (batch_size, num_outside_words, vec_dim)

        z_u = sigmoid(np.einsum("bon,bkn->bok", center_word_vectors, u, optimize="greedy"))
        # (batch_size, num_outside_words, (k+1)*num_outside_words)
        loss = -np.log(z_u).sum(axis=2)  # (batch_size, num_outside_words)
        delta = 1 - z_u  # (batch_size, num_outside_words, (k+1)*num_outside_words)

        grad_center_vec = -np.einsum("bok,bkn->bon", delta, u, optimize="greedy").sum(axis=1)

        grad_outside_vec = np.zeros_like(self.outside_word_vectors)  # (num_unique_words, vec_dim)
        grads = -np.einsum("bok,bon->bkn", delta, center_word_vectors, optimize="greedy")
        grads[:, num_outside_words:] *= -1  # (k+1)*num_outside_words, vec_dim)
        np.add.at(grad_outside_vec, indices, grads)

        return loss, grad_center_vec, grad_outside_vec
