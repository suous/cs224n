import json
from typing import List

import sentencepiece as spm
import torch

_default_word2id = {
    "<pad>": 0,  # Pad Token
    "<s>": 1,  # Start Token
    "</s>": 2,  # End Token
    "<unk>": 3,  # Unknown Token
}


def to_input_tensor(word_ids: List[List[int]]) -> torch.Tensor:
    """
    Convert list of sentences (word ids) into tensor with necessary padding for shorter sentences.

    Parameters
    ----------
    word_ids : List[List[int]]
        list of sentences, where each sentence is represented as a list of word indices (integers).

    Returns
    -------
    torch.Tensor
        tensor of (batch_size, max_sentence_length_in_batch) containing padded word ids.

    Examples
    --------
    >>> to_input_tensor([[1, 2, 3], [4, 5], [6]])
    tensor([[1, 2, 3], [4, 5, 0], [6, 0, 0]])
    """
    sentences_t = pad_sentences(word_ids)
    return torch.tensor(sentences_t, dtype=torch.long)


def pad_sentences(sentences, pad_token=_default_word2id["<pad>"]):
    """
    Pad list of sentences according to the longest sentence in the batch.

    Parameters
    ----------
    sentences : List[List[int]]
        list of sentences, where each sentence is represented as a list of word indices (integers).
    pad_token : int, optional
        padding token, by default _default_word2id["<pad>"]

    Returns
    -------
    List[List[int]]
        list of sentences where sentences shorter than the max length sentence are padded out with the pad_token,
        such that each sentences in the batch now has equal length.

    Examples
    --------
    >>> pad_sentences([[1, 2, 3], [4, 5], [6]], pad_token=0)
    [[1, 2, 3], [4, 5, 0], [6, 0, 0]]
    """
    max_len = max([len(sent) for sent in sentences])
    return [sent + [pad_token] * (max_len - len(sent)) for sent in sentences]


def collate_fn(batch):
    """
    Collate function to be used when wrapping a dataset in a DataLoader.

    Parameters
    ----------
    batch : List[Tuple[List[int], List[int]]]
        list of tuples of source and target sentences,
        where each sentence is represented as a list of word indices (integers).

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        tuple of tensors containing source and target sentences.

    Examples
    --------
    >>> collate_fn([([1, 2, 3], [4, 5]), ([6], [7, 8, 9])])
    (tensor([[1, 2, 3], [6, 0, 0]]), tensor([[4, 5, 0], [7, 8, 9]]))
    """
    src_sentences, tgt_sentences = zip(*batch)
    return to_input_tensor(src_sentences), to_input_tensor(tgt_sentences)


class VocabEntry(object):
    def __init__(self, word2id=None):
        self.word2id = word2id or _default_word2id
        self.id2word = {v: k for k, v in self.word2id.items()}

    def __getitem__(self, word):
        return self.word2id.get(word, self.word2id["<unk>"])

    def __contains__(self, word):
        return word in self.word2id

    def __setitem__(self, key, value):
        raise ValueError("vocabulary is readonly")

    def __len__(self):
        return len(self.word2id)

    def __repr__(self):
        return f"Vocabulary[size={len(self)}]"

    def id2word(self, wid):
        return self.id2word[wid]

    def add(self, word):
        if word in self:
            return self[word]
        wid = self.word2id[word] = len(self)
        self.id2word[wid] = word
        return wid

    def words2indices(self, sentences):
        return [[self[w] for w in s] for s in sentences]

    def indices2words(self, word_ids):
        return [self.id2word[w_id] for w_id in word_ids]


class Vocab(object):
    def __init__(self, src_vocab: VocabEntry, tgt_vocab: VocabEntry):
        self.src = src_vocab
        self.tgt = tgt_vocab

    @staticmethod
    def load(file_path):
        with open(file_path, "r") as f:
            entry = json.load(f)
            src_word2id = entry["src_word2id"]
            tgt_word2id = entry["tgt_word2id"]

        return Vocab(VocabEntry(src_word2id), VocabEntry(tgt_word2id))

    def __repr__(self):
        return f"Vocab(source {len(self.src)} words, target {len(self.tgt)} words)"


def read_corpus(file_path, source_path):
    data = []
    sp = spm.SentencePieceProcessor(model_file=str(source_path))

    with open(file_path, "r", encoding="utf8") as f:
        for line in f:
            subword_tokens = sp.encode_as_pieces(line)
            # only append <s> and </s> to the target sentence
            if source_path.stem == "tgt":
                subword_tokens = ["<s>"] + subword_tokens + ["</s>"]
            data.append(subword_tokens)

    return data
