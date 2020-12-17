import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import os
from torch import optim
from . import network
from transformers import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification, RobertaModel, RobertaTokenizer, RobertaForSequenceClassification


# ***************** Poolers **************************
# pooler's forwards accepts:
#   - base-encoding: output of the base encoding layer.
#     dimensions: [B*N*K, L, E], where L is sequence length
#     and E is size of base layer encoding.
#   - pos1, pos2: indexes of the head/tail entity [B*N*K]

# Default Bert pooler (first token)
class SingleTokenPooler(nn.Module):
    def __init__(self, base_layer_size, token_index=0):
        super().__init__()
        self.dense = nn.Linear(base_layer_size, base_layer_size)
        self.activation = nn.Tanh()
        self.token_index: int = token_index

    def forward(self, base_encoding, pos1, pos2):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = base_encoding[:, self.token_index]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class CatEntities(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, base_encoding, pos1, pos2):
        tensor_range = torch.arange(base_encoding.size()[0])
        h_state = base_encoding[tensor_range, pos1]
        t_state = base_encoding[tensor_range, pos2]
        state = torch.cat((h_state, t_state), -1)
        return state


def create_rep_layer(base_layer_size: int, strategy: str = "first") -> nn.Module:
    if strategy == "cls":
        return SingleTokenPooler(base_layer_size, 0)
    elif strategy == "cat_entity_reps":
        return CatEntities()
    else:
        raise NotImplementedError("Bad pooling strategy:" + strategy)

class ModularEncoder(nn.Module):
    def __init__(self, base_layer: nn.Module, base_layer_size: int, strategy: str):
        nn.Module.__init__(self)
        self.base = base_layer
        self.rep = create_rep_layer(base_layer_size, strategy)
    
    def forward(self, inputs):
        encoding = self.base(inputs["word"], inputs["mask"])
        representation = self.rep(encoding, inputs["pos1"], inputs["pos2"])
        return representation
    
    def tokenize(self, raw_tokens, pos_head, pos_tail, mask_entity=False):
        return self.base.tokenize(raw_tokens, pos_head, pos_tail, mask_entity)

class BERTBaseLayer(nn.Module):

    def __init__(self, pretrain_path, max_length): 
        nn.Module.__init__(self)
        self.bert = BertModel.from_pretrained(pretrain_path)
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained(pretrain_path)
    
    def forward(self, words, mask):
        return self.bert(words, mask)[0]
    
    def tokenize(self, raw_tokens, pos_head, pos_tail, mask_entity=False):
        # token -> index
        tokens = ['[CLS]']
        cur_pos = 0
        pos1_in_index = 1
        pos2_in_index = 1
        for token in raw_tokens:
            token = token.lower()
            if cur_pos == pos_head[0]:
                tokens.append('[unused0]')
                pos1_in_index = len(tokens)
            if cur_pos == pos_tail[0]:
                tokens.append('[unused1]')
                pos2_in_index = len(tokens)
            if mask_entity and ((pos_head[0] <= cur_pos and cur_pos <= pos_head[-1]) or (pos_tail[0] <= cur_pos and cur_pos <= pos_tail[-1])):
                tokens += ['[unused4]']
            else:
                tokens += self.tokenizer.tokenize(token)
            if cur_pos == pos_head[-1]:
                tokens.append('[unused2]')
            if cur_pos == pos_tail[-1]:
                tokens.append('[unused3]')
            cur_pos += 1
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        
        # padding
        while len(indexed_tokens) < self.max_length:
            indexed_tokens.append(0)
        indexed_tokens = indexed_tokens[:self.max_length]

        # pos
        pos1 = np.zeros((self.max_length), dtype=np.int32)
        pos2 = np.zeros((self.max_length), dtype=np.int32)
        for i in range(self.max_length):
            pos1[i] = i - pos1_in_index + self.max_length
            pos2[i] = i - pos2_in_index + self.max_length

        # mask
        mask = np.zeros((self.max_length), dtype=np.int32)
        mask[:len(tokens)] = 1

        pos1_in_index = min(self.max_length, pos1_in_index)
        pos2_in_index = min(self.max_length, pos2_in_index)

        return indexed_tokens, pos1_in_index - 1, pos2_in_index - 1, mask

class RobertaBaseLayer(nn.Module):

    def __init__(self, pretrain_path, max_length): 
        nn.Module.__init__(self)
        self.roberta = RobertaModel.from_pretrained(pretrain_path)
        self.max_length = max_length
        self.tokenizer = RobertaTokenizer.from_pretrained(pretrain_path)
    
    def forward(self, words, mask):
        return self.roberta(words, attention_mask=mask)[0]
    
    def tokenize(self, raw_tokens, pos_head, pos_tail, mask_entity=False):
        def getIns(bped, bpeTokens, tokens, L):
            resL = 0
            tkL = " ".join(tokens[:L])
            bped_tkL = " ".join(self.tokenizer.tokenize(tkL))
            if bped.find(bped_tkL) == 0:
                resL = len(bped_tkL.split())
            else:
                tkL += " "
                bped_tkL = " ".join(self.tokenizer.tokenize(tkL))
                if bped.find(bped_tkL) == 0:
                    resL = len(bped_tkL.split())
                else:
                    raise Exception("Cannot locate the position")
            return resL

        s = " ".join(raw_tokens)
        sst = self.tokenizer.tokenize(s)
        headL = pos_head[0]
        headR = pos_head[-1] + 1
        hiL = getIns(" ".join(sst), sst, raw_tokens, headL)
        hiR = getIns(" ".join(sst), sst, raw_tokens, headR)

        tailL = pos_tail[0]
        tailR = pos_tail[-1] + 1
        tiL = getIns(" ".join(sst), sst, raw_tokens, tailL)
        tiR = getIns(" ".join(sst), sst, raw_tokens, tailR)

        E1b = 'madeupword0000'
        E1e = 'madeupword0001'
        E2b = 'madeupword0002'
        E2e = 'madeupword0003'
        ins = [(hiL, E1b), (hiR, E1e), (tiL, E2b), (tiR, E2e)]
        ins = sorted(ins)
        pE1 = 0
        pE2 = 0
        pE1_ = 0
        pE2_ = 0
        for i in range(0, 4):
            sst.insert(ins[i][0] + i, ins[i][1])
            if ins[i][1] == E1b:
                pE1 = ins[i][0] + i
            elif ins[i][1] == E2b:
                pE2 = ins[i][0] + i
            elif ins[i][1] == E1e:
                pE1_ = ins[i][0] + i
            else:
                pE2_ = ins[i][0] + i
        pos1_in_index = pE1 + 1
        pos2_in_index = pE2 + 1
        sst = ['<s>'] + sst
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(sst)

        # padding
        while len(indexed_tokens) < self.max_length:
            indexed_tokens.append(1)
        indexed_tokens = indexed_tokens[:self.max_length]

        # pos
        pos1 = np.zeros((self.max_length), dtype=np.int32)
        pos2 = np.zeros((self.max_length), dtype=np.int32)
        for i in range(self.max_length):
            pos1[i] = i - pos1_in_index + self.max_length
            pos2[i] = i - pos2_in_index + self.max_length

        # mask
        mask = np.zeros((self.max_length), dtype=np.int32)
        mask[:len(sst)] = 1

        pos1_in_index = min(self.max_length, pos1_in_index)
        pos2_in_index = min(self.max_length, pos2_in_index)

        return indexed_tokens, pos1_in_index - 1, pos2_in_index - 1, mask

