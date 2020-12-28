from fewrel.fewshot_re_kit.data_loader import get_loader, get_loader_pair, get_loader_unsupervised
from fewrel.fewshot_re_kit.framework import FewShotREFramework
from fewrel.fewshot_re_kit.sentence_encoder import CNNSentenceEncoder
from fewrel.fewshot_re_kit.modular_encoder import ModularEncoder, BERTBaseLayer, LukeBaseLayer, RobertaBaseLayer
import fewrel.models
from fewrel.models.proto import Proto
from fewrel.models.gnn import GNN
from fewrel.models.snail import SNAIL
from fewrel.models.metanet import MetaNet
from fewrel.models.siamese import Siamese
from fewrel.models.pair import Pair
from fewrel.models.d import Discriminator
from fewrel.models.mtb import Mtb
import sys
import torch
from torch import optim, nn
import numpy as np
import json
import argparse
import os

def get_field(json_file, field_name):
    assert os.path.exists(json_file), "file not found: " + json_file
    with open(json_file, "r") as input:
        obj = json.load(input)
        assert field_name in obj, \
            "field '" + field_name + "' not found in json obj. Available fields are:\n\t" + str([str(key) for key in obj.keys()])
        return obj[field_name]

def get_encoder(encoder_name, pretrain_ckpt, max_length, strategy):
    if encoder_name == 'cnn':
        prefix = pretrain_ckpt or 'glove/glove'
        try:
            mat_file = './pretrain/' + prefix + '_mat.npy'
            glove_mat = np.load(mat_file)
        except:
            raise Exception("Cannot find non-contextual embedding file:", mat_file)
        try:
            index_file = './pretrain/'+ prefix + '_word2id.json'
            glove_word2id = json.load(open(index_file))
        except:
            raise Exception("Cannot find non-contextual embedding index file:", index_file)
        
        sentence_encoder = CNNSentenceEncoder(
                glove_mat,
                glove_word2id,
                max_length)
    elif encoder_name == 'bert':
        pretrain_ckpt = pretrain_ckpt or './pretrain/bert-base-uncased'
        sentence_encoder = ModularEncoder(
            BERTBaseLayer(pretrain_ckpt, max_length),
            get_field(os.path.join(pretrain_ckpt, "config.json"), "hidden_size"),
            strategy)
    elif encoder_name == 'roberta':
        pretrain_ckpt = pretrain_ckpt or './pretrain/roberta-base'
        sentence_encoder = ModularEncoder(
            RobertaBaseLayer(pretrain_ckpt, max_length),
            get_field(os.path.join(pretrain_ckpt, "config.json"), "hidden_size"),
            strategy)
    elif encoder_name == 'luke':
        pretrain_ckpt = pretrain_ckpt or './pretrain/luke'
        sentence_encoder = ModularEncoder(
            LukeBaseLayer(pretrain_ckpt, max_length),
            1024, # Hardcoded for now.
            strategy)
    else:
        raise NotImplementedError
    
    return sentence_encoder

def create_parser():
    parser = argparse.ArgumentParser()
    # Setup
    parser.add_argument('--data_root', default='./fewrel/data',
            help='path to the data folder')
    parser.add_argument('--train', default='train_wiki',
            help='train file')
    parser.add_argument('--val', default='val_wiki',
            help='val file')
    parser.add_argument('--test', default='test_wiki',
            help='test file')
    parser.add_argument('--adv', default=None,
            help='adv file')
    parser.add_argument('--trainN', default=10, type=int,
            help='N in train')
    parser.add_argument('--N', default=5, type=int,
            help='N way')
    parser.add_argument('--K', default=5, type=int,
            help='K shot')
    parser.add_argument('--Q', default=5, type=int,
            help='Num of query per class')
    parser.add_argument('--only_test', action='store_true',
           help='only test')
    
    # training params
    parser.add_argument('--batch_size', default=4, type=int,
            help='batch size')
    parser.add_argument('--train_iter', default=30000, type=int,
            help='num of iters in training')
    parser.add_argument('--val_iter', default=1000, type=int,
            help='num of iters in validation')
    parser.add_argument('--test_iter', default=1000, type=int,
            help='num of iters in testing')
    parser.add_argument('--val_step', default=2000, type=int,
           help='val after training how many iters')
    parser.add_argument('--max_length', default=128, type=int,
           help='max length')
    parser.add_argument('--lr', default=-1, type=float,
           help='learning rate')
    parser.add_argument('--weight_decay', default=1e-5, type=float,
           help='weight decay')
    parser.add_argument('--dropout', default=0.0, type=float,
           help='dropout rate')
    parser.add_argument('--na_rate', default=0, type=int,
           help='NA rate (NA = Q * na_rate)')
    parser.add_argument('--grad_iter', default=1, type=int,
           help='accumulate gradient every x iterations')
    parser.add_argument('--optim', default='sgd',
           help='sgd / adam / adamw')
    parser.add_argument('--fp16', action='store_true',
           help='use nvidia apex fp16')
    
    # model params
    parser.add_argument('--encoder', default='cnn',
            help='encoder: cnn or bert or roberta or luke')
    parser.add_argument('--pool', default='cls',
            choices=['cls', 'cat_entity_reps'])
    parser.add_argument('--hidden_size', default=230, type=int,
           help='hidden size')
    parser.add_argument('--load_ckpt', default=None,
           help='load ckpt')
    parser.add_argument('--save_ckpt', default=None,
           help='save ckpt')
    parser.add_argument('--ckpt_name', type=str, default='',
           help='checkpoint name.')

    # only for bert / roberta /luke
    parser.add_argument('--pretrain_ckpt', default=None,
           help='bert / roberta / luke pre-trained checkpoint')
    
    # only for prototypical networks
    parser.add_argument('--dot', action='store_true', 
           help='use dot instead of L2 distance for proto')

    # experiment
    parser.add_argument('--mask_entity', action='store_true',
           help='mask entity names')

    return parser

def main():
    
    torch.set_num_threads(20)

    opt = create_parser().parse_args()
    trainN = opt.trainN
    N = opt.N
    K = opt.K
    Q = opt.Q
    batch_size = opt.batch_size
    encoder_name = opt.encoder
    max_length = opt.max_length
    
    print("{}-way-{}-shot Few-Shot Relation Classification".format(N, K))
    print("encoder: {}".format(encoder_name))
    print("pooler: {}".format(opt.pool))
    print("max_length: {}".format(max_length))
    
    sentence_encoder = get_encoder(encoder_name, opt.pretrain_ckpt, max_length, opt.pool)

    train_data_loader = get_loader(opt.train, sentence_encoder,
            N=trainN, K=K, Q=Q, na_rate=opt.na_rate, batch_size=batch_size, root=opt.data_root)
    val_data_loader = get_loader(opt.val, sentence_encoder,
            N=N, K=K, Q=Q, na_rate=opt.na_rate, batch_size=batch_size, root=opt.data_root)
    test_data_loader = get_loader(opt.test, sentence_encoder,
            N=N, K=K, Q=Q, na_rate=opt.na_rate, batch_size=batch_size, root=opt.data_root)
    
    if opt.optim == 'sgd':
        pytorch_optim = optim.SGD
    elif opt.optim == 'adam':
        pytorch_optim = optim.Adam
    elif opt.optim == 'adamw':
        from transformers import AdamW
        pytorch_optim = AdamW
    else:
        raise NotImplementedError
    
    framework = FewShotREFramework(train_data_loader, val_data_loader, test_data_loader)
        
    prefix = '-'.join([encoder_name, opt.pool, opt.train, opt.val, str(N), str(K)])
    if opt.na_rate != 0:
        prefix += '-na{}'.format(opt.na_rate)
    if opt.dot:
        prefix += '-dot'
    if len(opt.ckpt_name) > 0:
        prefix += '-' + opt.ckpt_name
    
    multiple_gpu = True
    if encoder_name == "luke": # running luke model with multiple GPUs is buggy. 
        multiple_gpu = False
    model = Proto(sentence_encoder, dot=opt.dot, multiple_gpu=multiple_gpu)
    
    if not os.path.exists('checkpoint'):
        os.mkdir('checkpoint')
    ckpt = 'checkpoint/{}.pth.tar'.format(prefix)
    if opt.save_ckpt:
        ckpt = opt.save_ckpt

    if torch.cuda.is_available():
        model.cuda()

    if not opt.only_test:
        if encoder_name in ['bert', 'roberta', 'luke']:
            bert_optim = True
        else:
            bert_optim = False

        if opt.lr == -1:
            if bert_optim:
                opt.lr = 2e-5
            else:
                opt.lr = 1e-1

        framework.train(model, prefix, batch_size, trainN, N, K, Q,
                pytorch_optim=pytorch_optim, load_ckpt=opt.load_ckpt, save_ckpt=ckpt,
                na_rate=opt.na_rate, val_step=opt.val_step, fp16=opt.fp16, pair=False, 
                train_iter=opt.train_iter, val_iter=opt.val_iter, bert_optim=bert_optim, 
                learning_rate=opt.lr, use_sgd_for_bert=False)
    else:
        ckpt = opt.load_ckpt
        if ckpt is None:
            print("Warning: --load_ckpt is not specified. Will load Hugginface pre-trained checkpoint.")
            ckpt = 'none'

    acc = framework.eval(model, batch_size, N, K, Q, opt.test_iter, na_rate=opt.na_rate, ckpt=ckpt, pair=False)
    print("RESULT: %.2f" % (acc * 100))

def check_data():
    if not os.path.exists("./data"):
        base_dir = os.path.dirname(__file__)
        os.system("ln -s {} .".format(os.path.join(base_dir, "data")))
    if not os.path.exists("./pretrain"):
        os.system("wget http://libnlp.s3.dev.obdc.bcs.bloomberg.com/data/fewrel/pretrain.tar -o dummy")
        os.system("tar -xvf pretrain.tar")    

if __name__ == "__main__":
    check_data()
    main()
