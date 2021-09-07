#!/usr/bin/env python3

###############################################################################
# This file was added to the original FewRel repository as part of the work
# described in the paper "Towards Realistic Few-Shot Relation Extraction",
# published in EMNLP 2021.
#
# It contains an alternate evaluation strategy that mimics a binary relation
# extraction scenario, on a per-relation basis.
# 
# Authors: Sam Brody (sbrody18@bloomberg.net), Sichao Wu (swu389@bloomberg.net),
#          Adrian Benton (abenton10@bloomberg.net)
###############################################################################


import argparse
import json
import os
import random
import sys
from statistics import mean, stdev

import numpy as np
import torch
from torch import nn, optim

import fewrel.models
from fewrel.fewshot_re_kit.data_loader import (
    get_loader,
    get_loader_pair,
    get_loader_unsupervised,
    load_data_for_alt_eval,
)
from fewrel.fewshot_re_kit.framework import FewShotREFramework
from fewrel.fewshot_re_kit.modular_encoder import (
    BERTBaseLayer,
    LukeBaseLayer,
    ModularEncoder,
    RobertaBaseLayer,
)
from fewrel.fewshot_re_kit.sentence_encoder import CNNSentenceEncoder
from fewrel.models.d import Discriminator
from fewrel.models.gnn import GNN
from fewrel.models.metanet import MetaNet
from fewrel.models.mtb import Mtb
from fewrel.models.pair import Pair
from fewrel.models.proto import Proto
from fewrel.models.siamese import Siamese
from fewrel.models.snail import SNAIL
from fewrel.util import get_encoder


def create_parser():
    parser = argparse.ArgumentParser()
    # Setup
    parser.add_argument('--test', default='test_wiki',
            help='test file')
    parser.add_argument('--data_root', default='./fewrel/data',
            help='path to the data folder')
    parser.add_argument('--K', default=5, type=int,
            help='Number of example used for prototype')
    parser.add_argument('--T', default=50, type=int,
            help='Total number of examples to include in test set (per relation)')
        
    # model params
    parser.add_argument(
        "--encoder",
        default="cnn",
        help="encoder",
        choices=["cnn", "bert", "spanbert", "roberta", "luke"],
    )
    parser.add_argument("--pool", default="cls", choices=["cls", "cat_entity_reps"])
    parser.add_argument("--hidden_size", default=230, type=int, help="hidden size")
    parser.add_argument(
        "--pretrain_ckpt",
        default=None,
        help="bert / spanbert / roberta / luke pre-trained checkpoint",
    )
    parser.add_argument("--load_ckpt", default=None, help="load ckpt")
    parser.add_argument("--max_length", default=128, type=int, help="max length")

    # only for prototypical networks
    parser.add_argument(
        "--dot", action="store_true", help="use dot instead of L2 distance for proto"
    )

    # experiment
    parser.add_argument("--mask_entity", action="store_true", help="mask entity names")
    parser.add_argument(
        "--no_entity_token",
        action="store_true",
        help="whether to remove special tokens before and after entities",
    )
    parser.add_argument(
        "--num_threads", type=int, default=20, help="number of threads for PyTorch"
    )
    parser.add_argument(
        "--num_iter", type=int, default=10, help="number of evaluation iterations"
    )

    return parser


def calc_auc(scored_labels, true_label):
    """Calculates area under the (precision-recall) curve."""
    tp = 0
    total = 0
    for (
        _,
        l,
    ) in scored_labels:
        if l == true_label:
            tp += 1
        else:
            total += tp
    return float(total) / (tp * (len(scored_labels) - tp))


def calc_confusion(scored_labels, true_label, size):
    """Calculates the confusion matrix."""
    conf = {}

    for _, l in scored_labels[:n]:
        if l not in conf:
            conf[l] = 0
        conf[l] += 1.0 / n
    return conf


def eval(model, rel, K, instances, label_to_idxs, batch_size=100):
    confusion = None
    # select random K, average them
    sample = random.sample(range(*label_to_idxs[rel]), k=K)
    proto = torch.stack([instances[i] for i in sample]).mean(0)
    scores = []
    # calc distance from each instance (batched)
    for b in range(0, len(instances), batch_size):
        e = min(b + batch_size, len(instances))
        dist = model.__dist__(proto, torch.stack(instances[b:e]), dim=1)
        scores.extend(dist.tolist())

    # calc auc
    labels = [""] * len(instances)
    for r in label_to_idxs.keys():
        b, e = label_to_idxs[r]
        labels[b:e] = [r] * (e - b)

    scored_labels = list(zip(scores, labels))
    scored_labels.sort(key=lambda x: x[0], reverse=True)

    auc = calc_auc(scored_labels, rel)
    confusion = calc_confusion(
        scored_labels, rel, label_to_idxs[rel][1] - label_to_idxs[rel][0]
    )

    return auc, confusion


def main():
    opt = create_parser().parse_args()

    torch.set_num_threads(opt.num_threads)

    K = opt.K
    encoder_name = opt.encoder
    max_length = opt.max_length

    assert os.path.exists(opt.pretrain_ckpt)
    if opt.load_ckpt:
        assert os.path.exists(opt.load_ckpt)

    print(f"{K}-shot Few-Shot Relation AUC Eval")
    print(f"encoder: {encoder_name}")
    print(f"pooler: {opt.pool}")
    print(f"max_length: {max_length}")

    sentence_encoder = get_encoder(
        encoder_name,
        opt.pretrain_ckpt,
        max_length,
        opt.pool,
        opt.load_ckpt,
        mask_entity=opt.mask_entity,
        add_entity_token=not opt.no_entity_token,
    )

    instances, label_to_idxs = load_data_for_alt_eval(opt.test, sentence_encoder,
            num_instances=opt.T, batch_size=100, root=opt.data_root)

    model = Proto(sentence_encoder, dot=opt.dot)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    conf_matrix = {rel: {r: 0 for r in label_to_idxs} for rel in label_to_idxs}
    stats = {}
    for rel in label_to_idxs:
        results = [
            eval(model, rel, K, instances, label_to_idxs) for i in range(opt.num_iter)
        ]
        aucs, confs = list(zip(*results))
        for conf in confs:
            for r in conf:
                conf_matrix[rel][r] += conf[r] / opt.num_iter

        print(rel)
        stats[rel] = {"mean": mean(aucs), "stdev": stdev(aucs)}
        print(round(stats[rel]["mean"], 2), round(stats[rel]["stdev"], 3), "\n")

    print("----\t", "\t".join(list(label_to_idxs.keys())))
    for r in label_to_idxs:
        print(
            r,
            "\t",
            "\t".join([str(round(conf_matrix[r][c], 3)) for c in label_to_idxs]),
        )

    if opt.load_ckpt:
        suffix = f"{os.path.splitext(os.path.basename(opt.load_ckpt))[0]}-{opt.test.replace('/', '_')}"
    else:
        suffix = f"{os.path.basename(os.path.normpath(opt.pretrain_ckpt))}-{opt.pool}-{opt.test.replace('/', '_')}"
    with open(f"stats_{suffix}.json", "w") as outfile:
        json.dump(stats, outfile)
    with open(f"confusion_{suffix}.json", "w") as outfile:
        json.dump(conf_matrix, outfile)


if __name__ == "__main__":
    main()
