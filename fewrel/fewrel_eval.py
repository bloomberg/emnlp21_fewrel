#!/usr/bin/env python3

###############################################################################
# This file was ported from the original FewRel repository as part of the work
# described in the paper "Towards Realistic Few-Shot Relation Extraction",
# published in EMNLP 2021.
#
# It contains a code for training and evaluating models in the original FewRel
# setup.
# 
# Authors: Sam Brody (sbrody18@bloomberg.net), Sichao Wu (swu389@bloomberg.net),
#          Adrian Benton (abenton10@bloomberg.net)
###############################################################################


import argparse
import json
import os
import sys

import numpy as np
import torch
from torch import nn, optim
from transformers import AdamW

import fewrel.models
from fewrel.fewshot_re_kit.data_loader import (
    get_loader,
    get_loader_pair,
    get_loader_unsupervised,
)
from fewrel.fewshot_re_kit.framework import FewShotREFramework
from fewrel.fewshot_re_kit.modular_encoder import (
    BERTBaseLayer,
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
    parser.add_argument(
        "--data_root", default="./fewrel/data", help="path to the data folder"
    )
    parser.add_argument("--train", default="train_wiki", help="train file")
    parser.add_argument("--val", default="val_wiki", help="val file")
    parser.add_argument("--test", default="test_wiki", help="test file")
    parser.add_argument("--adv", default=None, help="adv file")
    parser.add_argument("--trainN", default=10, type=int, help="N in train")
    parser.add_argument("--N", default=5, type=int, help="N way")
    parser.add_argument("--K", default=5, type=int, help="K shot")
    parser.add_argument("--Q", default=5, type=int, help="Num of query per class")
    parser.add_argument("--only_test", action="store_true", help="only test")

    # training params
    parser.add_argument("--batch_size", default=4, type=int, help="batch size")
    parser.add_argument(
        "--train_iter", default=30000, type=int, help="num of iters in training"
    )
    parser.add_argument(
        "--val_iter", default=1000, type=int, help="num of iters in validation"
    )
    parser.add_argument(
        "--test_iter", default=1000, type=int, help="num of iters in testing"
    )
    parser.add_argument(
        "--val_step", default=2000, type=int, help="val after training how many iters"
    )
    parser.add_argument("--max_length", default=128, type=int, help="max length")
    parser.add_argument("--lr", default=-1, type=float, help="learning rate")
    parser.add_argument("--weight_decay", default=1e-5, type=float, help="weight decay")
    parser.add_argument("--dropout", default=0.0, type=float, help="dropout rate")
    parser.add_argument(
        "--na_rate", default=0, type=int, help="NA rate (NA = Q * na_rate)"
    )
    parser.add_argument(
        "--grad_iter",
        default=1,
        type=int,
        help="accumulate gradient every x iterations",
    )
    parser.add_argument("--optim", default="sgd", help="sgd / adam / adamw")
    parser.add_argument("--fp16", action="store_true", help="use nvidia apex fp16")
    parser.add_argument("--num_threads", default=20, type=int, help="number of threads for Pytorch")
    
    # model params
    parser.add_argument(
        "--encoder",
        default="cnn",
        help="encoder: cnn or bert or spanbert or roberta or luke",
    )
    parser.add_argument("--pool", default="cls", choices=["cls", "cat_entity_reps"])
    parser.add_argument("--hidden_size", default=230, type=int, help="hidden size")
    parser.add_argument("--load_ckpt", default=None, help="load ckpt")
    parser.add_argument("--save_ckpt", default=None, help="save ckpt")
    parser.add_argument("--ckpt_name", type=str, default="", help="checkpoint name")

    # only for bert / spanbert/ roberta /luke
    parser.add_argument(
        "--pretrain_ckpt",
        default=None,
        help="bert / spanbert/ roberta / luke pre-trained checkpoint",
    )

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

    return parser


def main():

    opt = create_parser().parse_args()
    torch.set_num_threads(opt.num_threads)

    trainN = opt.trainN
    N = opt.N
    K = opt.K
    Q = opt.Q
    batch_size = opt.batch_size
    encoder_name = opt.encoder
    max_length = opt.max_length

    print(f"{N}-way-{K}-shot Few-Shot Relation Classification")
    print(f"encoder: {encoder_name}")
    print(f"pooler: {opt.pool}")
    print(f"max_length: {max_length}")

    sentence_encoder = get_encoder(
        encoder_name,
        opt.pretrain_ckpt,
        max_length,
        opt.pool,
        mask_entity=opt.mask_entity,
        add_entity_token=not opt.no_entity_token,
    )

    train_data_loader = get_loader(
        opt.train,
        sentence_encoder,
        N=trainN,
        K=K,
        Q=Q,
        na_rate=opt.na_rate,
        batch_size=batch_size,
        root=opt.data_root,
    )
    val_data_loader = get_loader(
        opt.val,
        sentence_encoder,
        N=N,
        K=K,
        Q=Q,
        na_rate=opt.na_rate,
        batch_size=batch_size,
        root=opt.data_root,
    )
    test_data_loader = get_loader(
        opt.test,
        sentence_encoder,
        N=N,
        K=K,
        Q=Q,
        na_rate=opt.na_rate,
        batch_size=batch_size,
        root=opt.data_root,
    )

    OPTIM = {
        "sgd": optim.SGD,
        "adam": optim.Adam,
        "adamw": AdamW
    }
    pytorch_optim = OPTIM[opt.optim]

    framework = FewShotREFramework(train_data_loader, val_data_loader, test_data_loader)

    prefix = "-".join([encoder_name, opt.pool, opt.train, opt.val, str(N), str(K)])
    if opt.na_rate != 0:
        prefix += f"-na{opt.na_rate}"
    if opt.dot:
        prefix += "-dot"
    if len(opt.ckpt_name) > 0:
        prefix += "-" + opt.ckpt_name
    if opt.mask_entity:
        prefix += "-mask_entity"
    if opt.no_entity_token:
        prefix += "-no_entity_token"

    multiple_gpu = True
    if encoder_name == "luke":  # Original luke implementation doesn't work well with multiple GPUs.
        multiple_gpu = False
    model = Proto(sentence_encoder, dot=opt.dot, multiple_gpu=multiple_gpu)

    if not os.path.exists("checkpoint"):
        os.mkdir("checkpoint")
    ckpt = f"checkpoint/{prefix}.pth.tar"
    if opt.save_ckpt:
        ckpt = opt.save_ckpt

    if torch.cuda.is_available():
        model.cuda()

    if not opt.only_test:
        if encoder_name in ["bert", "spanbert", "roberta", "luke"]:
            bert_optim = True
        else:
            bert_optim = False

        if opt.lr == -1:
            if bert_optim:
                opt.lr = 2e-5
            else:
                opt.lr = 1e-1

        framework.train(
            model,
            prefix,
            batch_size,
            trainN,
            N,
            K,
            Q,
            pytorch_optim=pytorch_optim,
            load_ckpt=opt.load_ckpt,
            save_ckpt=ckpt,
            na_rate=opt.na_rate,
            val_step=opt.val_step,
            fp16=opt.fp16,
            pair=False,
            train_iter=opt.train_iter,
            val_iter=opt.val_iter,
            bert_optim=bert_optim,
            learning_rate=opt.lr,
            use_sgd_for_bert=False,
        )
    else:
        ckpt = opt.load_ckpt
        if ckpt is None:
            print(
                "Warning: --load_ckpt is not specified. Will load Hugginface pre-trained checkpoint."
            )
            ckpt = "none"

    acc = framework.eval(
        model,
        batch_size,
        N,
        K,
        Q,
        opt.test_iter,
        na_rate=opt.na_rate,
        ckpt=ckpt,
        pair=False,
    )
    print("RESULT: %.2f" % (acc * 100))


if __name__ == "__main__":
    main()
