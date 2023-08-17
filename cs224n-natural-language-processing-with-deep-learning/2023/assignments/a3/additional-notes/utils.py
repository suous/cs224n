import itertools
from collections import Counter

import pandas as pd

P_PREFIX = "<p>:"
L_PREFIX = "<l>:"
UNK = "<UNK>"
NULL = "<NULL>"
ROOT = "<ROOT>"


def build_dict(keys, n_max=None, offset=0):
    count = Counter(keys)
    return {w: i + offset for i, (w, _) in enumerate(count.most_common(n=n_max))}


class Parser:
    # ruff: noqa: PLR0913
    def __init__(self, df, with_punc=True, unlabeled=True, lowercase=True, use_pos=True, use_dep=True):
        self.with_punc = with_punc
        self.unlabeled = unlabeled
        self.lowercase = lowercase
        self.use_pos = use_pos
        self.use_dep = use_dep and (not unlabeled)

        # Find the index of the root head in each sentence
        df["root_index"] = df["head"].apply(lambda x: x.index(0))
        # Extract the label of the root word in each sentence
        df["root_labels"] = df.apply(lambda x: x.label[x.root_index], axis=1)

        assert df["root_labels"].nunique() == 1, "There should be only one root label"
        self.root_label = df["root_labels"].unique()[0]

        unique_labels = pd.Series(list(set(itertools.chain.from_iterable(df.label))))
        tok2id = dict(zip(L_PREFIX + unique_labels, range(len(unique_labels))))
        tok2id[L_PREFIX + NULL] = self.L_NULL = len(tok2id)

        # Add pos tags to the dictionary
        tok2id.update(build_dict([P_PREFIX + w for w in itertools.chain.from_iterable(df.pos)], offset=len(tok2id)))
        tok2id[P_PREFIX + UNK] = self.P_UNK = len(tok2id)
        tok2id[P_PREFIX + NULL] = self.P_NULL = len(tok2id)
        tok2id[P_PREFIX + ROOT] = self.P_ROOT = len(tok2id)

        # Add word tokens to the dictionary
        tok2id.update(build_dict(itertools.chain.from_iterable(df.word), offset=len(tok2id)))
        tok2id[UNK] = self.UNK = len(tok2id)
        tok2id[NULL] = self.NULL = len(tok2id)
        tok2id[ROOT] = self.ROOT = len(tok2id)

        self.tok2id = tok2id
        self.id2tok = {v: k for (k, v) in tok2id.items()}
        self.n_features = 18 + (18 if self.use_pos else 0) + (12 if self.use_dep else 0)
        self.n_tokens = len(tok2id)

        trans = ["L", "R", "S"]
        self.num_unique_labels = 1

        if not unlabeled:
            trans = ("L-" + unique_labels).to_list() + ("R-" + unique_labels).to_list() + ["S"]
            self.num_unique_labels = len(unique_labels)

        self.n_trans = len(trans)
        self.tran2id = {t: i for (i, t) in enumerate(trans)}
        self.id2tran = dict(enumerate(trans))

    def vectorize(self, *examples):
        vec_examples = []
        for ex in examples:
            word = [self.ROOT] + [self.tok2id.get(w, self.UNK) for w in ex["word"]]
            pos = [self.P_ROOT] + [self.tok2id.get(P_PREFIX + w, self.P_UNK) for w in ex["pos"]]
            head = [-1] + ex["head"]
            label = [-1] + [self.tok2id.get(L_PREFIX + w, -1) for w in ex["label"]]
            vec_examples.append({"word": word, "pos": pos, "head": head, "label": label})
        return vec_examples

    # ruff: noqa: PLR0911
    def get_oracle(self, stack, buf, ex):
        minimal_stacks = 2
        if len(stack) < minimal_stacks:
            return self.n_trans - 1

        # extracts the indices, heads, and labels of the top two words on the stack.
        [i1, i0] = stack[-2:]
        h0, h1 = ex["head"][i0], ex["head"][i1]
        l0, l1 = ex["label"][i0], ex["label"][i1]

        if self.unlabeled:
            if i1 > 0 and h1 == i0:
                return 0
            elif 0 <= i1 == h0 and not any(x for x in buf if ex["head"][x] == i0):
                return 1
            else:
                return None if len(buf) == 0 else 2

        if i1 > 0 and h1 == i0:
            return l1 if (l1 >= 0) and (l1 < self.num_unique_labels) else None
        elif 0 <= i1 == h0 and not any(x for x in buf if ex["head"][x] == i0):
            return l0 + self.num_unique_labels if (l0 >= 0) and (l0 < self.num_unique_labels) else None
        else:
            return None if len(buf) == 0 else self.n_trans - 1

    def legal_labels(self, stack, buf):
        # 1 if the stack has more than 2 elements, 0 otherwise
        # 1 if the stack has at least 2 elements, 0 otherwise
        # 1 if the buffer has more than 0 elements, 0 otherwise
        minimal_stacks = 2
        return (
            [int(len(stack) > minimal_stacks)] * self.num_unique_labels
            + [int(len(stack) >= minimal_stacks)] * self.num_unique_labels
            + [int(len(buf) > 0)]
        )

    def extract_features(self, stack, buf, arcs, ex):
        def get_lc(k):
            return sorted([arc[1] for arc in arcs if arc[0] == k and arc[1] < k])

        def get_rc(k):
            return sorted([arc[1] for arc in arcs if arc[0] == k and arc[1] > k], reverse=True)

        def get_features(key, sentinel, window_size=3):
            return (
                [sentinel] * (window_size - len(stack))
                + [ex[key][x] for x in stack[-window_size:]]
                + [ex[key][x] for x in buf[:window_size]]
                + [sentinel] * (window_size - len(buf))
            )

        p_features = []
        l_features = []
        features = get_features("word", self.NULL)
        if self.use_pos:
            p_features = get_features("pos", self.P_NULL)

        num_sentinels = 6
        for i in range(2):
            if i >= len(stack):
                features += [self.NULL] * num_sentinels
                if self.use_pos:
                    p_features += [self.P_NULL] * num_sentinels
                if self.use_dep:
                    l_features += [self.L_NULL] * num_sentinels
                continue

            k = stack[-i - 1]
            lc = get_lc(k)
            rc = get_rc(k)
            llc = get_lc(lc[0]) if len(lc) > 0 else []
            rrc = get_rc(rc[0]) if len(rc) > 0 else []

            features.append(ex["word"][lc[0]] if len(lc) > 0 else self.NULL)
            features.append(ex["word"][rc[0]] if len(rc) > 0 else self.NULL)
            features.append(ex["word"][lc[1]] if len(lc) > 1 else self.NULL)
            features.append(ex["word"][rc[1]] if len(rc) > 1 else self.NULL)
            features.append(ex["word"][llc[0]] if len(llc) > 0 else self.NULL)
            features.append(ex["word"][rrc[0]] if len(rrc) > 0 else self.NULL)

            if self.use_pos:
                p_features.append(ex["pos"][lc[0]] if len(lc) > 0 else self.P_NULL)
                p_features.append(ex["pos"][rc[0]] if len(rc) > 0 else self.P_NULL)
                p_features.append(ex["pos"][lc[1]] if len(lc) > 1 else self.P_NULL)
                p_features.append(ex["pos"][rc[1]] if len(rc) > 1 else self.P_NULL)
                p_features.append(ex["pos"][llc[0]] if len(llc) > 0 else self.P_NULL)
                p_features.append(ex["pos"][rrc[0]] if len(rrc) > 0 else self.P_NULL)

            if self.use_dep:
                l_features.append(ex["label"][lc[0]] if len(lc) > 0 else self.L_NULL)
                l_features.append(ex["label"][rc[0]] if len(rc) > 0 else self.L_NULL)
                l_features.append(ex["label"][lc[1]] if len(lc) > 1 else self.L_NULL)
                l_features.append(ex["label"][rc[1]] if len(rc) > 1 else self.L_NULL)
                l_features.append(ex["label"][llc[0]] if len(llc) > 0 else self.L_NULL)
                l_features.append(ex["label"][rrc[0]] if len(rrc) > 0 else self.L_NULL)

        features += p_features + l_features
        assert len(features) == self.n_features
        return features

    def create_instances(self, *examples):
        all_instances = []
        for ex in examples:
            n_words = len(ex["word"]) - 1

            stack = [0]
            buf = list(range(1, n_words + 1))
            arcs = []
            instances = []
            for _ in range(n_words * 2):
                gold_t = self.get_oracle(stack, buf, ex)
                if gold_t is None:
                    break
                legal_labels = self.legal_labels(stack, buf)
                assert legal_labels[gold_t] == 1
                instances.append((self.extract_features(stack, buf, arcs, ex), legal_labels, gold_t))
                if gold_t == self.n_trans - 1:
                    # SHIFT: move the first element of the buffer to the stack
                    stack.append(buf.pop(0))
                elif gold_t < self.num_unique_labels:
                    # LEFT-ARC
                    # 1. generate a dependency tuple head -> dependent: (stack[-1], stack[-2])
                    # 2. remove the second element of the stack
                    arcs.append((stack[-1], stack.pop(-2), gold_t))
                else:
                    # RIGHT-ARC
                    # 1. generate a dependency tuple head -> dependent: (stack[-2], stack[-1])
                    # 2. remove the first element of the stack
                    arcs.append((stack[-2], stack.pop(-1), gold_t - self.num_unique_labels))
            all_instances += instances

        return all_instances


def read_conll(in_file, lowercase=False, max_example=None, num_fields=10):
    with open(in_file) as f:
        word, pos, head, label = [], [], [], []
        for i, line in enumerate(f):
            sp = line.strip().split("\t")
            if len(sp) == num_fields and "-" not in sp[0]:
                word.append(sp[1].lower() if lowercase else sp[1])
                pos.append(sp[4])
                head.append(int(sp[6]))
                label.append(sp[7])
            elif word:
                yield {"word": word, "pos": pos, "head": head, "label": label}
                word, pos, head, label = [], [], [], []
                if max_example is not None and i >= max_example:
                    break
        if word:
            yield {"word": word, "pos": pos, "head": head, "label": label}
