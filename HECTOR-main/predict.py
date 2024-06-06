import argparse
from collections import defaultdict
import configparser
from datetime import datetime

import numpy as np

from data_loader import Batch, build_label_paths, build_subsequent_mask, create_dataloaders, \
    load_tokenizer, load_vocab, load_ontology, load_taxonomy, load_dataset
from model import load_trained_model

import torch


if torch.cuda.is_available():
    dev = "cuda"
else:
    dev = "cpu"
device = torch.device(dev)


def sort_by_key(input, index):
    """Sort input by index"""
    index = index.unsqueeze(2).expand_as(input)
    return torch.gather(input, 1, index)


def beam_search(model, src, src_mask, prefixes, level_range, beam_width, vocab_tgt, level2lid):
    """
    Implement beam search algorithm starting from the given prefixes
    :param model: model instance
    :param src: torch.tensor (batch_size, src_seq_len); src sequence
    :param src_mask: torch.tensor (batch_size, 1, src_seq_len); src sequence mask
    :param prefixes: list[list[int]]; list of prefixes for each instance, each prefix is a list of label idx
    :param level_range: tuple(int, int); min_level – from which decoding starts, max_level – taxonomy depth
    :param beam_width: int; beam width
    :param vocab_tgt: torchtext.vocab.Vocab; mapping between labels and their ids
    :param level2lid: {int: list[int]}; key – level, value – list of label idx of this level
    :return: torch.tensor (batch_size, tgt_vocab_size); predicted probabilities
    """
    start_idx = vocab_tgt["<s>"]
    batch_size = src.shape[0]
    min_level, max_level = level_range

    # final scores matrix
    output = torch.zeros(batch_size, len(vocab_tgt)).fill_(-1e9).cuda()  # final scores are negative => default -inf

    queue = []
    nprefixes = [len(prefixes_i) for prefixes_i in prefixes]  # number of prefixes per instance
    Tnprefixes = torch.tensor(nprefixes, device="cuda")
    nprefixes_cumsum = np.cumsum([0] + nprefixes)

    Tprefixes = []
    Tscores = []

    for prefixes_i in prefixes:  # prefixes of the i-th instance
        for prefix in prefixes_i:
            Tprefixes.append(torch.tensor([start_idx] + prefix).cuda())  # prepend each prefix with <s>
            Tscores.append(torch.zeros(len(prefix) + 1).cuda())  # init scores of prefix labels with 0

    queue.append((torch.stack(Tprefixes, dim=0), torch.stack(Tscores, dim=0)))

    # repeat each i-th row of src and src_mask nprefixes[i] times
    src = torch.repeat_interleave(src, Tnprefixes, dim=0)  # bs_expand x src_seq_len
    src_mask = torch.repeat_interleave(src_mask, Tnprefixes, dim=0)   # bs_expand x 1 x src_seq_len

    memory = model.encode(src, src_mask)  # bs_expand x src_seq_len x src_dim
    memory = model.adapter(memory)  # bs_expand x src_seq_len x tgt_dim

    for level in range(min_level, max_level + 1):
        new_prefixes = []
        new_scores = []
        level_label_ids = level2lid[level]  # labels of the current level

        for prefixes, scores in queue:  # prefixes: bs_expand x level; prefix_scores: bs_expand x level
            out = model.decode(
                memory=memory,
                src_mask=src_mask,
                tgt=prefixes,
                tgt_mask=build_subsequent_mask(prefixes.size(-1)).type_as(src.data)
            )  # bs_expand x level x tgt_dim

            # individual label probs for the last element in the decoded sequence
            out = model.generator(out)[:, -1, :]  # bs_expand x vocab_size

            # joint label probs (w.r.t. the prefix) for the last element in the decoded sequence
            joint_prob = out + scores[:, -1].unsqueeze(1)  # bs_expand x vocab_size

            for i in range(batch_size):
                if nprefixes_cumsum[i] == nprefixes_cumsum[i + 1]:  # instance has 0 prefixes
                    continue

                # rows from nprefixes_cumsum[i] to nprefixes_cumsum[i + 1] represent one instance
                # for each instance, take the maximum joint probability for each label across all prefixes
                max_prob = torch.max(joint_prob[nprefixes_cumsum[i]:nprefixes_cumsum[i + 1], :], dim=0)[0]

                # assign max_prob scores to the labels of the current level and save them to the final scores matrix
                output[i, level_label_ids] = torch.maximum(output[i, level_label_ids], max_prob[level_label_ids])

            # append next label to each prefix
            # we select top-k labels, where k == beam_width
            prob_sorted, indices = torch.sort(joint_prob, descending=True)
            for i in range(beam_width):
                prefixes_ext = torch.cat([prefixes, indices[:, i].unsqueeze(1)], dim=1)  # bs_expand x (level + 1)
                prefix_scores_ext = torch.cat([scores, prob_sorted[:, i].unsqueeze(1)], dim=1)  # bs_expand x (level + 1)
                new_prefixes.append(prefixes_ext)
                new_scores.append(prefix_scores_ext)

        new_prefixes = torch.stack(new_prefixes, dim=1)  # bs_expand x beam^2 x (level + 1)
        new_scores = torch.stack(new_scores, dim=1)  # bs_expand x beam^2 x path_len

        # prefix score is a joint probability of all labels from this prefix
        # score of each label is a joint probability of all prior labels => score of the last label == prefix score
        new_prefix_scores = new_scores[:, :, -1]
        new_prefix_scores_sorted, new_prefix_scores_indices = torch.sort(new_prefix_scores, dim=1, descending=True)

        # sort new prefixes by their scores
        new_prefixes_sorted = sort_by_key(new_prefixes, new_prefix_scores_indices)
        new_scores_sorted = sort_by_key(new_scores, new_prefix_scores_indices)

        queue = []
        for j in range(beam_width):
            queue.append((new_prefixes_sorted[:, j, :], new_scores_sorted[:, j, :]))

    return output


def run_prediction(config, model_path, start_level):
    """
    Load data, load model and run beam search (batch mode)
    :param config: config
    :param model_path: path to the pretrained model
    :param start_level: level from which we start refinement
    :return: tuple(predictions, scores); each element is an np.array of size (len(testset), 1000)
    """
    print(f"{datetime.now()} Loading data")
    testset = load_dataset(config["Paths"]["test"])

    print(f"{len(testset)} test instances")

    # Tokenizer for source sequence (text in natural language)
    spacy_en = load_tokenizer()
    tokenizer_src = lambda x: [tok.text for tok in spacy_en.tokenizer(x)]

    label2level = load_ontology(config["Paths"]["ontology"])
    parent2children = load_taxonomy(config["Paths"]["taxonomy"])

    # Each vocab is instance of torchtext.vocab.Vocab, which maps tokens/labels to their unique ids
    vocab_src, vocab_tgt = load_vocab(config, tokenizer_src, label2level)

    # {int: list[int]} – list of label ids at each level
    level2lid = defaultdict(list)
    for label, level in label2level.items():
        level2lid[level].append(vocab_tgt[label])

    print(f"{datetime.now()} Loading model")
    model = load_trained_model(config, model_path, vocab_src, vocab_tgt)
    model.cuda(device)

    _, _, test_dataloader = create_dataloaders(
        device,
        vocab_src,
        vocab_tgt,
        tokenizer_src,
        config,
        mode="test"
    )
    data_iter = (Batch(b[0], b[1], vocab_src["<blank>"]) for b in test_dataloader)

    max_level = config["Prediction"].getint("max_level")
    batch_size = config["DataLoader"].getint("batch_size_test")

    print(f"{datetime.now()} Start predicting")
    all_pred = []
    all_scores = []

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(data_iter):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(testset))

            targets = []
            prefixes = []
            for j in range(start_idx, end_idx):
                targets_j = [label for label in testset[j][1] if label2level[label] >= start_level]
                prior = [label for label in testset[j][1] if label2level[label] < start_level]

                prefixes_j = build_label_paths(set(prior), [], config["DataLoader"]["root"], parent2children)
                prefixes_j = [p for p in prefixes_j if len(p) == start_level - 1]  # keep only prefixes up to start_level
                prefixes_j_idx = [vocab_tgt.lookup_indices(prefix) for prefix in prefixes_j]  # convert labels to ids

                targets.append(targets_j)
                prefixes.append(prefixes_j_idx)

            # scores for each label (batch_size x vocab_size)
            output = beam_search(
                model,
                batch.src,
                batch.src_mask,
                prefixes=prefixes,
                level_range=(start_level, max_level),
                beam_width=config["Prediction"].getint("beam"),
                vocab_tgt=vocab_tgt,
                level2lid=level2lid
            )

            top_scores, top_pred = torch.topk(output, k=1000, dim=1)  # keep top-k predicted labels

            for row in top_pred:
                labels = vocab_tgt.lookup_tokens(row.cpu().tolist())  # convert ids to labels
                all_pred.append(np.array([labels], dtype=object))
            all_scores.append(top_scores.cpu().numpy())

            print(f"{datetime.now()} {(i + 1) * batch_size} instances processed")

    all_pred = np.concatenate(all_pred, axis=0)
    all_scores = np.concatenate(all_scores, axis=0)

    return all_pred, all_scores


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", default="config_MAG.ini")
    parser.add_argument("--model", help="Model to evaluate")
    parser.add_argument("--level", type=int, help="from which level we start predictions")
    parser.add_argument("--output", help="Path to save results")

    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config)

    print(f"{datetime.now()} Starting experiment: model: {args.model}, level: {args.level}, output path: {args.output}")

    predictions, scores = run_prediction(config, args.model, args.level)
    np.save(args.output + "-labels.npy", predictions)
    np.save(args.output + "-scores.npy", scores)


if __name__ == '__main__':
    main()


