import argparse
import json

import numpy as np


def compute_precision_at(pred, tgt, k):
    """Compute prec@k for a single sample."""
    tp = 0
    for j in range(k):
        if pred[j] in tgt:
            tp += 1

    return tp / k


def compute_ndcg(pred, tgt, k):
    """Compute NDCG@k for a single sample."""
    log = 1.0 / np.log2(np.arange(k) + 2)
    dcg = 0
    for j in range(k):
        if pred[j] in tgt:
            dcg += log[j]
    ndcg = dcg / log.cumsum()[np.minimum(len(tgt), k) - 1]
    return ndcg


def update_metrics(metrics, pred, tgt):
    metrics["prec@1"] += compute_precision_at(pred, tgt, 1)
    metrics["prec@3"] += compute_precision_at(pred, tgt, 3)
    metrics["prec@5"] += compute_precision_at(pred, tgt, 5)
    metrics["ndcg@3"] += compute_ndcg(pred, tgt, 3)
    metrics["ndcg@5"] += compute_ndcg(pred, tgt, 5)


def run_eval(testset_path, pred_path, onto_path, start_level):
    """
    Evaluate the quality of model's predictions
    The following metrics are calculated:
    - prec@1 == ndcg@1
    - prec@3
    - prec@5
    - ndcg@3
    - ndcg@5
    Only labels of level >= start_level are considered (both in target and in predictions)
    """
    relevant_labels = set()
    with open(onto_path) as fin:
        for line in fin:
            data = json.loads(line)
            if data["level"] >= start_level:
                relevant_labels.add(data["label"])

    predictions = np.load(pred_path, allow_pickle=True)
    testset = [json.loads(line) for line in open(testset_path)]

    assert predictions.shape[0] == len(testset)

    metrics = {
        "prec@1": 0,
        "prec@3": 0,
        "prec@5": 0,
        "ndcg@3": 0,
        "ndcg@5": 0,
    }
    total = 0

    for i in range(predictions.shape[0]):
        pred_i = [l for l in predictions[i] if l in relevant_labels]
        tgt_i = [l for l in testset[i]["label"] if l in relevant_labels]

        if len(tgt_i) == 0:
            continue

        update_metrics(metrics, pred_i, tgt_i)
        total += 1

    print("Final results: ")
    print("prec@1: {}, prec@3: {}, prec@5: {}, nDCG@3: {}, nDCG@5: {}".format(
        metrics["prec@1"] / total,
        metrics["prec@3"] / total,
        metrics["prec@5"] / total,
        metrics["ndcg@3"] / total,
        metrics["ndcg@5"] / total,
    ))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--testset", help="Path to the test set", required=True)
    parser.add_argument("--pred", help="Path to the numpy array with predicted labels", required=True)
    parser.add_argument("--ontology", help="Path to the ontology", required=True)
    parser.add_argument("--level", type=int, help="From which level prediction starts", default=1)

    args = parser.parse_args()

    run_eval(args.testset, args.pred, args.ontology, args.level)


if __name__ == '__main__':
    main()
