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

    def get_random_context(self, num_context=5):
        while True:
            sentence = random.choice(self.sentences)
            word_idx = random.randint(0, len(sentence) - 1)
            center_word = sentence[word_idx]
            context = sentence[max(0, word_idx - num_context) : min(len(sentence), word_idx + num_context)]
            context = [w for w in context if w != center_word]
            if len(context) > 0:
                return center_word, context

    def get_negative_samples(self, outside_word_index, k=10):
        negative_sample_word_index = [0] * k
        for i in range(k):
            while idx := self.sample_token_index:
                if idx != outside_word_index:
                    break
            negative_sample_word_index[i] = idx
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

    def _train(self, center_word, outside_words):
        grad_center_word_vectors = np.zeros_like(self.center_word_vectors)
        grad_outside_word_vectors = np.zeros_like(self.outside_word_vectors)

        center_word_index = self.word2index[center_word]
        center_word_vector = self.center_word_vectors[center_word_index]

        loss = 0.0
        for outside_word_index in (self.word2index[w] for w in outside_words):
            _loss, _grad_center_word_vectors, _grad_outside_word_vectors = self._forward(
                center_word_vectors=center_word_vector, outside_word_index=outside_word_index
            )
            loss += _loss
            grad_center_word_vectors[center_word_index] += _grad_center_word_vectors
            grad_outside_word_vectors += _grad_outside_word_vectors

        return loss, grad_center_word_vectors, grad_outside_word_vectors

    def train_one_epoch(self, batch_size):
        center_word_grad = np.zeros_like(self.center_word_vectors)
        outside_word_grad = np.zeros_like(self.outside_word_vectors)

        loss = 0.0
        for _i in range(batch_size):
            num_context = random.randint(1, self.num_context)
            center_word, context = self.dataset.get_random_context(num_context=num_context)

            _loss, grad_center_word, grad_outside_word = self._train(
                center_word=center_word,
                outside_words=context,
            )
            loss += _loss / batch_size
            center_word_grad += grad_center_word / batch_size
            outside_word_grad += grad_outside_word / batch_size

        return loss, center_word_grad, outside_word_grad

    def naive_softmax_loss_and_gradient(
        self,
        center_word_vectors,
        outside_word_index,
    ):
        """
        Naive Softmax loss & gradient function for word2vec models.

        Parameters
        ----------
        center_word_vectors : np.ndarray, shape=(1,vec_dim)
            The center word vectors.
        outside_word_index : int or list
            The index of the outside word.

        Returns
        -------
        loss : float
            The softmax loss.
        grad_center_vec : np.ndarray, shape=(1,vec_dim)
            The gradient of the center word vector.
        grad_outside_vec : np.ndarray, shape=(num_outside_words,vec_dim)
            The gradient of the outside word vectors.
        """

        # v_c ->  center_word_vectors:    (n,)    ->   (1,n)
        # o   ->  outside_word_index:     (1,)    ->   (1,1)
        # U   ->  outside_word_vectors:   (m,n)
        # y   ->  target:                 (m,)    ->   (1,m)
        # z   ->  np.dot(v_c, U.T):       (m,)    ->   (1,m)

        if isinstance(outside_word_index, int):
            outside_word_index = [outside_word_index]

        if center_word_vectors.ndim == 1:
            center_word_vectors = center_word_vectors.reshape((1, -1))  # (1,n)

        z = center_word_vectors @ self.outside_word_vectors.T  # (m,1)
        u_o = self.outside_word_vectors[outside_word_index]  # (1,n)
        loss = np.log(np.exp(z).sum()) - center_word_vectors @ u_o.T  # scalar

        y_hat = softmax(z)  # (1,m)
        y = np.zeros_like(y_hat)  # (1,m)
        y[:, outside_word_index] = 1  # (1,m)

        grad_center_vec = (y_hat - y) @ self.outside_word_vectors  # (1,n)
        grad_outside_vec = (y_hat - y).T @ center_word_vectors  # (m,n)

        return loss.squeeze(), grad_center_vec.squeeze(), grad_outside_vec

    def negative_sampling_loss_and_gradient(self, center_word_vectors, outside_word_index, k=10):
        """
        Negative sampling loss & gradient function for word2vec models.

        Parameters
        ----------
        center_word_vectors : np.ndarray, shape=(1,vec_dim)
            The center word vectors.
        outside_word_index : int or list
            The index of the outside word.
        k : int, default=10
            The number of negative samples.

        Returns
        -------
        loss : float
            The negative sampling loss.
        grad_center_vec : np.ndarray, shape=(1,vec_dim)
            The gradient of the center word vector.
        grad_outside_vec : np.ndarray, shape=(num_outside_words,vec_dim)
            The gradient of the outside word vectors.
        """

        negative_sample_indices = self.dataset.get_negative_samples(outside_word_index, k=k)  # (k,)

        if isinstance(outside_word_index, int):
            outside_word_index = [outside_word_index]

        if center_word_vectors.ndim == 1:
            center_word_vectors = center_word_vectors.reshape((1, -1))  # (1,n)

        indices = outside_word_index + negative_sample_indices  # (k+1,)

        u_o = self.outside_word_vectors[outside_word_index]  # (1,n)
        u_n = self.outside_word_vectors[negative_sample_indices]  # (K,n)

        u = np.vstack((u_o, -u_n))  # (K+1,n)

        z_u = sigmoid(center_word_vectors @ u.T)  # (1,K+1)
        loss = -np.log(z_u).sum()  # scalar
        delta = 1 - z_u  # (1,K+1)

        grad_center_vec = -(delta @ u).squeeze()  # (1,n)

        grad_outside_vec = np.zeros_like(self.outside_word_vectors)  # (m,n)
        grads = delta.T @ center_word_vectors  # (K+1,n)
        grads[0] *= -1  # (K+1,n)
        np.add.at(grad_outside_vec, indices, grads)

        return loss, grad_center_vec, grad_outside_vec
