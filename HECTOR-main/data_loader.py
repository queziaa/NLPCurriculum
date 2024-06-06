from collections import OrderedDict
import json
import logging
import os
import random

from gensim.models import KeyedVectors
import numpy as np

import spacy
import torch
from torch.nn.functional import pad
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator, vocab


# ----------------------------------------------------
# Load data from disk
# ----------------------------------------------------

def load_dataset(path):
    """
    Load X (src document) and Y (relevant labels)
    :param path: path to the dataset file
    :return: list of tuples [(str(x), list(y))]
    """
    dataset = []

    with open(path) as fin:
        for line in fin:
            data = json.loads(line)
            dataset.append((data["text_processed"], data["label"]))

    return dataset


def load_taxonomy(path):
    """
    Load label taxonomy <parent : children>
    :param path: path to the taxonomy file
    :return: dict {parent_label: [children_labels]}
    """
    parent2children = {}
    with open(path) as fin:
        for line in fin:
            labels = line.strip().split(" ")
            parent2children[labels[0]] = labels[1:]
    return parent2children


def load_ontology(path):
    """
    Build <label : level> mapping from ontology
    :param path: path to the ontology file
    :return: dict {label: level}
    """
    label2level = {}
    with open(path) as fin:
        for line in fin:
            data = json.loads(line)
            label2level[data["label"]] = data["level"]
    return label2level


def load_tokenizer():
    "Load spaCy tokenizer for English language."
    try:
        spacy_en = spacy.load("en_core_web_sm")
    except IOError:
        os.system("python -m spacy download en_core_web_sm")
        spacy_en = spacy.load("en_core_web_sm")

    return spacy_en


# ----------------------------------------------------
# Build token and label vocabularies
# ----------------------------------------------------

def load_vocab(config, tokenizer_src, label2level):
    """
    Load token and label vocabularies from config["Paths"]["vocab"] file.
    If the file does not exist – build vocabularies and save them as tuple to config["Paths"]["vocab"].
    :param config: config
    :param tokenizer_src: SpaCy tokenizer for natural language sequences
    :param label2level: dict {label: level}
    :return: tuple of torchtext.vocab.Vocab, where each Vocab is a mapping between tokens/labels and their ids
    """
    if not os.path.exists(config["Paths"]["vocab"]):
        dataset = load_dataset(config["Paths"]["train"]) + load_dataset(config["Paths"]["valid"])
        vocab_src = build_token_vocab(dataset, tokenizer_src, min_freq=config["DataLoader"].getint("min_freq"))
        vocab_tgt = build_label_vocab(dataset, label2level, min_freq=1)
        torch.save((vocab_src, vocab_tgt), config["Paths"]["vocab"])
    else:
        vocab_src, vocab_tgt = torch.load(config["Paths"]["vocab"])
    logging.info(f"Vocabulary sizes: tokens: {len(vocab_src)}, labels: {len(vocab_tgt)}")
    return vocab_src, vocab_tgt


def build_token_vocab(dataset, tokenizer_src, min_freq):
    """
    Build token vocabulary: <token : unique ID>
    :param dataset: list of tuples [(str(x), str(y))], where x is a text and y is a sequence of labels
    :param min_freq: min frequency of a token to be included into the vocabulary
    :param tokenizer_src: SpaCy tokenizer for natural language sequence
    :return: torchtext.vocab.Vocab, mapping between tokens and their ids
    """

    logging.info("Building src vocabulary ...")

    vocab_src = build_vocab_from_iterator(
        (tokenizer_src(x) for x, _ in dataset),
        min_freq=min_freq,
        specials=["<blank>", "<s>", "</s>", "<unk>"],
    )

    #  Default index for an unknown token
    vocab_src.set_default_index(vocab_src["<unk>"])

    return vocab_src


def build_label_vocab(dataset, label2level, min_freq):
    """
    Build label vocabulary: <label : unique ID>
    IDs are ordered by label level id(label_level_1) < id(label_level_2) < id(label_level_3)...
    :param dataset: list of tuples [(str(x), str(y))], where x is a text and y is a sequence of labels
    :param label2level: dict {label: level}
    :param min_freq: min frequency of a token to be included into the vocabulary
                    (we want all labels in the vocabulary => min_freq == 1)
    :return: torchtext.vocab.Vocab, mapping between labels and their ids
    """
    logging.info("Building tgt vocabulary ...")
    ordered_labels = order_labels_by_level(dataset, label2level)
    vocab_tgt = vocab(
        ordered_labels,
        min_freq=min_freq,
        specials=["<blank>", "<s>", "</s>", "<unk>"]
    )

    #  Default index will be returned when OOV token is queried
    vocab_tgt.set_default_index(vocab_tgt["<unk>"])

    return vocab_tgt


def order_labels_by_level(dataset, label2level):
    """
    The order in which key-value pairs are inserted in the label vocabulary is important
    (we want that that labels of 1st level to go before labels of 2nd level etc.)
    This function build an OrderedDict object, where keys are labels and values are frequency
    (always 1, because we don't care about label frequency)
    torchtext.vocab.vocab factory method takes an OrderedDict as an argument and insert keys to the vocabulary
    w.r.t. the ordered in which they were inserted in OrderedDict
    :param dataset: list of tuples [(str(x), str(y))], where x is a text and y is a sequence of labels
    :param label2level: dict {label: level}
    :return: OrderedDict with ordered labels {label: 1}
    """
    labels_w_level = []

    for _, labels in dataset:
        for label in labels:
            labels_w_level.append((label, label2level[label]))

    labels_w_level = list(set(labels_w_level))  # remove duplicates
    labels_w_level = sorted(labels_w_level, key=lambda x: x[1])
    labels_ordered = [l for l, _ in labels_w_level]

    labels_ordered_dict = OrderedDict.fromkeys(labels_ordered, value=1)

    return labels_ordered_dict

# ----------------------------------------------------
# Create data loaders
# ----------------------------------------------------


def create_dataloaders(device, vocab_src, vocab_tgt, tokenizer_src, config, mode="train"):
    """
    Create Python iterables over train, valid and test datasets
    :param device: cuda or gpu
    :param vocab_src: instance of torchtext.vocab.Vocab, mapping between tokens and their ids
    :param vocab_tgt: instance of torchtext.vocab.Vocab, mapping between labels and their ids
    :param tokenizer_src: SpaCy tokenizer
    :param config: config
    :param mode: ["train", "test"]
    :return: tuple of torch.utils.data.DataLoader (train, valid, test)
             when mode == train: (train, valid, None)
             when mode == test: (None, None, test)
    """

    # wrapper collate_batch() function (see below)
    def collate_fn(batch):
        return collate_batch(
            batch,
            tokenizer_src,
            vocab_src,
            vocab_tgt,
            device,
            max_padding_src,
            max_padding_tgt,
        )

    if mode == "train":
        batch_size = config["DataLoader"].getint("batch_size_train")

        train_data = load_dataset(config["Paths"]["train"])  # [doc, [labels]]
        valid_data = load_dataset(config["Paths"]["valid"])  # [doc, [labels]]

        taxo = load_taxonomy(config["Paths"]["taxonomy"])
        taxo_root = config["DataLoader"]["root"]

        # convert a set of labels to the set of paths

        # for train_iter, each instance has multiple paths
        # at each training epoch a random path is selected (see collate_batch())
        train_iter = []
        for x, y in train_data:
            y_paths = build_label_paths(set(y), [], taxo_root, taxo)  # [[path1], [path2], ...]
            train_iter.append((x, y_paths))

        # for valid_iter, each instance has a single path (for deterministic evaluation)
        valid_iter = []
        for x, y in valid_data:
            y_paths = build_label_paths(set(y), [], taxo_root, taxo)  # [[path1], [path2], ...]
            for path in y_paths:
                valid_iter.append((x, [path]))

        max_padding_src = config["DataLoader"].getint("max_padding_src")
        max_padding_tgt = config["DataLoader"].getint("max_padding_tgt")

        train_dataloader = DataLoader(
            train_iter,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )
        valid_dataloader = DataLoader(
            valid_iter,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )
        test_dataloader = None

    elif mode == "test":
        batch_size = config["DataLoader"].getint("batch_size_test")

        test_data = load_dataset(config["Paths"]["test"])

        # collate_fn expects tgt of type list[list[str]]
        test_iter = [(x, [y]) for x, y in test_data]

        max_padding_src = config["DataLoader"].getint("max_padding_src")
        max_padding_tgt = 100

        train_dataloader = None
        valid_dataloader = None
        test_dataloader = DataLoader(
            test_iter,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )
    else:
        raise Exception("Unknown dataloader type")

    return train_dataloader, valid_dataloader, test_dataloader


def build_label_paths(label_set, current_path, current_label, taxonomy):
    """
    Given a set of labels, organize them in paths
    Recursively traverse a taxonomy tree starting from the root, visiting only nodes from the label set
    :param label_set: initial set of labels
    :param current_path: a path from the root constructed so far
    :param current_label: current label
    :param taxonomy: {parent_label: [children_labels]} mapping
    :return list of lists, where each list represent a path
    """
    if len(label_set) == 0:
        return [current_path]

    if current_label not in taxonomy:
        return [current_path]

    children = set(taxonomy[current_label])
    if len(label_set.intersection(children)) == 0:
        return [current_path]

    paths = []
    for label in label_set:
        if label in taxonomy[current_label]:
            new_path = current_path + [label]
            new_label_set = label_set - {label}
            paths.extend(build_label_paths(new_label_set, new_path, label, taxonomy))

    return paths


def collate_batch(batch, tokenizer_src, vocab_src, vocab_tgt, device, max_padding_src, max_padding_tgt):
    """
    A function to collate a list of samples
    :param batch: list of (src, tgt) pairs, where
            src is a string of input document,
            tgt is a list of label paths list[list[str]]
    :param tokenizer_src: SpaCy tokenizer for src sequence
    :param vocab_src: instance of torchtext.vocab.Vocab, mapping between tokens and their ids
    :param vocab_tgt: instance of torchtext.vocab.Vocab, mapping between labels and their ids
    :param device: cuda or gpu
    :param max_padding_src: max number of tokens in src
    :param max_padding_tgt: max number of labels in tgt
    :return: a pair of (src, tgt) 2D tensors
    """

    Tbos = torch.tensor([vocab_src["<s>"]], device=device)  # tensor with <s> token id
    Teos = torch.tensor([vocab_src["</s>"]], device=device)  # tensor with </s> token id
    pad_id = vocab_src["<blank>"]  # <blank> token id

    src_list, tgt_list = [], []
    for (_src, _tgts) in batch:
        """
        Process src and tgt:
            1.1. select a random path from tgts
            1.2. Tokenize
            1.3. Convert tokens/labels to ids
            1.4. Prepend <s> token id, append </s> token id
            1.5. Pad
            1.6. Convert to tensor
        """
        _tgt = random.choice(_tgts)

        Tsrc = torch.tensor(
            vocab_src(tokenizer_src(_src)),
            dtype=torch.int64,
            device=device,
        )

        Ttgt = torch.tensor(
            vocab_tgt(_tgt),
            dtype=torch.int64,
            device=device,
        )

        Tsrc_enclosed = torch.cat([Tbos, Tsrc, Teos], 0)
        Ttgt_enclosed = torch.cat([Tbos, Ttgt, Teos], 0)

        Tsrc_padded = pad(
            Tsrc_enclosed,
            (0, max_padding_src - len(Tsrc_enclosed)),
            value=pad_id
        )
        Ttgt_padded = pad(
            Ttgt_enclosed,
            (0, max_padding_tgt - len(Ttgt_enclosed)),
            value=pad_id
        )

        src_list.append(Tsrc_padded)
        tgt_list.append(Ttgt_padded)

    # Concatenates a sequence of tensors along a new dimension
    src = torch.stack(src_list)
    tgt = torch.stack(tgt_list)

    return src, tgt


class Batch:
    """Object for holding a batch of data with mask during training."""
    def __init__(self, src, tgt=None, pad=0):  # 0 = <blank>
        """
        :param src: 2D tensor of shape (batch_size, max_padding_src) with document tokens ids
                    each row starts with 1 (<s> token id) and ends with 2 (</s> token id)
        :param tgt: 2D tensor of shape (batch_size, max_padding_tgt) with relevant labels ids
                    each row starts with 1 (<s> token id) and ends with 2 (</s> token id)
        :param pad: id of a padding token
        """
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)  # padding mask (non-masked positions: True, masked: False)
        if tgt is not None:
            # input for decoder: [<s>, labe1_0, label_1, ... label_m]
            # for decoding label_i, decoder looks at all prev labels [<s>, labe1_0, ..., label_i-1]
            self.tgt = tgt[:, :-1]

            # input for loss function is shifted left w.r.t. decoder input: [labe1_0, label_1, ... label_m, </s>]
            # from sequence [<s>, labe1_0, label_1] decoder should generate sequence [labe1_0, label_1, </s>]
            self.tgt_y = tgt[:, 1:]

            # padding mask & subsequent position mask (non-masked positions: True, masked: False)
            tgt_mask = (self.tgt != pad).unsqueeze(-2)
            subsequent_mask = build_subsequent_mask(self.tgt.size(-1)).type_as(tgt_mask.data)
            self.tgt_mask = tgt_mask & subsequent_mask

            # total number of relevant labels in the batch (for loss value normalization)
            self.ntokens = (self.tgt_y != pad).data.sum()


# ----------------------------------------------------
# Misc
# ----------------------------------------------------

def build_subsequent_mask(size):
    """Mask out subsequent positions."""
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def build_level_mask(vocab_tgt, label2level, config):
    seq_len = config["DataLoader"].getint("taxonomy_depth") + 1  # 1 - </s> label
    mask = torch.zeros(seq_len, len(vocab_tgt))

    for i, label in enumerate(vocab_tgt.get_itos()):
        level = label2level.get(label, 0)

        if level > 0:
            mask[level - 1, i] = 1
        else:  # special labels are possible at each position
            for j in range(seq_len):
                mask[j, i] = 1
    return mask


def build_init_embeddings(config, vocab_src):
    """
    Initialize token embeddings and save them to the file.
    We use GloVe as our initial embeddings; if a token does not have a GloVe embedding, it's initialized randomly
    :param config: config
    :param vocab_src: torchtext.vocab.Vocab; mapping between tokens and their ids
    :return: None (save the embedding matrix to file)
    """
    print("Loading GloVe embeddings")
    glove_emb = config["Paths"]["glove_model"]
    w2v_model = KeyedVectors.load(glove_emb)

    print("Building embedding src")
    emb_src_size = config["Model"].getint("d_src")
    pad = vocab_src["<blank>"]

    # check that GloVe embedding size corresponds to emb_src_size specified in config
    assert w2v_model["the"].shape[0] == emb_src_size

    emb_src_init = []
    for word in vocab_src.get_itos():
        if word == pad:
            emb = np.zeros(emb_src_size)
        elif word in w2v_model:
            emb = w2v_model[word]
        else:
            emb = np.random.uniform(-1.0, 1.0, emb_src_size)
        emb_src_init.append(emb)

    emb_init = np.asarray(emb_src_init)
    np.save(config["Paths"]["embedding_src"], emb_init)


def get_level_sizes(max_level, vocab_tgt, label2level):
    nlevels = max_level + 1
    level_sizes = [0 for _ in range(nlevels)]  # level 0 for OOV labels (eg. special labels <s>, <pad>, etc.)
    for label in vocab_tgt.get_itos():
        level = label2level.get(label, 0)
        level_sizes[level] += 1

    return level_sizes
