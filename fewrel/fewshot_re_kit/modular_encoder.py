###############################################################################
# This file was added to the original FewRel repository as part of the work
# described in the paper "Towards Realistic Few-Shot Relation Extraction",
# published in EMNLP 2021.
#
# It contains a more modular implementation of the logic in sentence_encoder.py,
# separating the encoder and pooler layers for easier mix-and-match functionality.
# 
# Authors: Sam Brody (sbrody18@bloomberg.net), Sichao Wu (swu389@bloomberg.net),
#          Adrian Benton (abenton10@bloomberg.net)
###############################################################################


import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from transformers import (
    BertForMaskedLM,
    BertForSequenceClassification,
    BertModel,
    BertTokenizer,
    RobertaForSequenceClassification,
    RobertaModel,
    RobertaTokenizer,
)

from . import network
from .luke.model import LukeEntityAwareAttentionModel, LukeModel
from .luke.utils.entity_vocab import HEAD_TOKEN, MASK_TOKEN, TAIL_TOKEN
from .luke.utils.model_utils import ModelArchive

# ***************** Poolers **********************************
#  Layers for converting the output of the encoder layer
#  to a representation for the input sequence.
#
#  The pooler's forward method accepts:
#   - base-encoding: output of the base encoding layer.
#     dimensions: [B*N*K, L, E], where L is sequence length
#     and E is size of base layer encoding.
#   - pos1, pos2: indexes of the head/tail entity [B*N*K]
# ************************************************************

class SingleTokenPooler(nn.Module):
    """Pooler which uses the encoding of a single token (default: first token) as the representation."""
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
    """Pooler which uses the concatenated encoding of two tokens (prefixing the entities) as the representation."""
    def __init__(self):
        super().__init__()

    def forward(self, base_encoding, pos1, pos2):
        tensor_range = torch.arange(base_encoding.size()[0])
        h_state = base_encoding[tensor_range, pos1]
        t_state = base_encoding[tensor_range, pos2]
        state = torch.cat((h_state, t_state), -1)
        return state


def create_rep_layer(base_layer_size: int, strategy: str = "cls") -> nn.Module:
    """Helper method which returns the correct pooler layer based on specified strategy."""    
    if strategy == "cls":
        return SingleTokenPooler(base_layer_size, 0)
    elif strategy == "cat_entity_reps":
        return CatEntities()
    else:
        raise NotImplementedError("Bad pooling strategy:" + strategy)

# ***************** Encoders ****************************************
#  Layers for converting the input text into a context-
#  aware vector representation.
#
#  The encoder has two methods:
#   - tokenize() which takes the raw input text and start positions
#     of the subject and object entities, and creates the necessary
#     input for the forward method.
#   - forward() with takes the output of tokenize() and returns the
#     vector representations for the tokens.
# *******************************************************************

def _pad_inputs(indexed_tokens, max_length, padding_val):
    """Helper method to pad or truncate indexed_tokens to the correct length."""
    while len(indexed_tokens) < max_length:
        indexed_tokens.append(padding_val)
    # truncate if too long
    indexed_tokens = indexed_tokens[:max_length]

def _create_positional_vectors(max_length, pos1_in_index, pos2_in_index):
    """Helper method to create positional vectors.
    The vectors contain a value of 'max_length' in the cell corresponding to the specified
    index, and values decrease (increase) to the left (right).
    """ 
    pos1 = np.zeros((max_length), dtype=np.int32)
    pos2 = np.zeros((max_length), dtype=np.int32)
    for i in range(max_length):
        pos1[i] = i - pos1_in_index + max_length
        pos2[i] = i - pos2_in_index + max_length
    return pos1, pos2


class BERTBaseLayer(nn.Module):
    """Implementation of the base encoder layer using BERT."""
    def __init__(self, pretrain_path, max_length, mask_entity=False, add_entity_token=True): 
        nn.Module.__init__(self)
        self.bert = BertModel.from_pretrained(pretrain_path)
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained(pretrain_path)
        self.mask_entity = mask_entity
        self.add_entity_token = add_entity_token

    def forward(self, words, mask):
        return self.bert(words, mask)[0]

    def tokenize(self, raw_tokens, pos_head, pos_tail):
        # token -> index
        tokens = ["[CLS]"]
        cur_pos = 0
        pos1_in_index = 1
        pos2_in_index = 1

        # Converts the raw text tokens into a sequence of indexes.
        # In the process, inserts special tokens before and after the entities.
        # if 'self.mask_entity' is true, also replaces the entity tokens with a
        # special mask token.
        for token in raw_tokens:
            token = token.lower()
            if cur_pos == pos_head[0]:
                pos1_in_index = len(tokens)
                if self.add_entity_token:
                    tokens.append("[unused0]")
            if cur_pos == pos_tail[0]:
                pos2_in_index = len(tokens)
                if self.add_entity_token:
                    tokens.append("[unused1]")
            if self.mask_entity and (
                (pos_head[0] <= cur_pos and cur_pos <= pos_head[-1])
                or (pos_tail[0] <= cur_pos and cur_pos <= pos_tail[-1])
            ):
                tokens += ["[unused4]"]
            else:
                tokens += self.tokenizer.tokenize(token)
            if cur_pos == pos_head[-1] and self.add_entity_token:
                tokens.append("[unused2]")
            if cur_pos == pos_tail[-1] and self.add_entity_token:
                tokens.append("[unused3]")
            cur_pos += 1
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)

        # padding
        _pad_inputs(indexed_tokens, self.max_length, 0)

        # positional vectors
        pos1, pos2 = _create_positional_vectors(self.max_length, pos1_in_index, pos2_in_index)

        # mask
        mask = np.zeros((self.max_length), dtype=np.int32)
        mask[: len(tokens)] = 1

        pos1_in_index = min(self.max_length, pos1_in_index)
        pos2_in_index = min(self.max_length, pos2_in_index)

        return indexed_tokens, pos1_in_index, pos2_in_index, mask


class RobertaBaseLayer(nn.Module):
    def __init__(
        self, pretrain_path, max_length, mask_entity=False, add_entity_token=True
    ):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(pretrain_path)
        self.max_length = max_length
        self.tokenizer = RobertaTokenizer.from_pretrained(pretrain_path)
        self.mask_entity = mask_entity
        self.add_entity_token = add_entity_token

    def forward(self, words, mask):
        return self.roberta(words, attention_mask=mask)[0]

    def tokenize(self, raw_tokens, pos_head, pos_tail):
        def getIns(bped, tokens, L):
            """Attempts to find the index of the L-th token in the byte-pair sequence.
            
            Args:
                bped: the byte-pair encoding of the entire text (all tokens).
                tokens: the original text tokens.
                L: the index of the token of interest.            
            """
            # Create the byte-pair encoding of the string up to the L-th token.
            resL = 0
            tkL = " ".join(tokens[:L])
            bped_tkL = " ".join(self.tokenizer.tokenize(tkL))

            if bped.find(bped_tkL) == 0:
                resL = len(bped_tkL.split())
            else:
                # try again with added space (in case token is mid-byte-pair).
                tkL += " "
                bped_tkL = " ".join(self.tokenizer.tokenize(tkL))
                if bped.find(bped_tkL) == 0:
                    resL = len(bped_tkL.split())
                else:
                    raise Exception("Cannot locate the position")
            return resL

        # Tokenize the raw text.
        s = " ".join(raw_tokens)
        sst = self.tokenizer.tokenize(s)
        
        # Get the head and tail indexes in the byte-pair encoding.
        headL = pos_head[0]
        headR = pos_head[-1] + 1
        hiL = getIns(" ".join(sst), raw_tokens, headL)
        hiR = getIns(" ".join(sst), raw_tokens, headR)

        tailL = pos_tail[0]
        tailR = pos_tail[-1] + 1
        tiL = getIns(" ".join(sst), raw_tokens, tailL)
        tiR = getIns(" ".join(sst), raw_tokens, tailR)

        # Add special tokens before and after the entities.
        E1b = 'madeupword0000'
        E1e = 'madeupword0001'
        E2b = 'madeupword0002'
        E2e = 'madeupword0003'
        # Create a list of locations and the tokens that should be inserted there.

        ins = [(hiL, E1b), (hiR, E1e), (tiL, E2b), (tiR, E2e)]
        ins = sorted(ins)

        # Pointers for the final locations of the entities.
        pE1 = hiL
        pE2 = tiL
        pE1_ = hiR
        pE2_ = tiR
        if self.add_entity_token:
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
        if self.mask_entity:
            # If special token is inserted before entity, do not mask that.
            pE1_start = pE1 + 1 if self.add_entity_token else pE1
            pE2_start = pE2 + 1 if self.add_entity_token else pE2
            sst[pE1_start:pE1_] = ["<mask>"] * (pE1_ - pE1_start)
            sst[pE2_start:pE2_] = ["<mask>"] * (pE2_ - pE2_start)
        sst = ["<s>"] + sst
        # Advance the index by one because of '<s>' is inserted
        pos1_in_index = pE1 + 1
        pos2_in_index = pE2 + 1

        indexed_tokens = self.tokenizer.convert_tokens_to_ids(sst)

        # padding
        _pad_inputs(indexed_tokens, self.max_length, 1)

        # positional vectors
        pos1, pos2 = _create_positional_vectors(self.max_length, pos1_in_index, pos2_in_index)

        # mask
        mask = np.zeros((self.max_length), dtype=np.int32)
        mask[: len(sst)] = 1

        pos1_in_index = min(self.max_length, pos1_in_index)
        pos2_in_index = min(self.max_length, pos2_in_index)

        return indexed_tokens, pos1_in_index, pos2_in_index, mask

class LukeBaseLayer(nn.Module):
    def __init__(
        self, pretrain_path, max_length, mask_entity=False, add_entity_token=True
    ):
        super().__init__()
        # Load pretrained LUKE model
        model_archive = ModelArchive.load(pretrain_path)
        self.tokenizer = model_archive.tokenizer
        self.max_mention_length = model_archive.max_mention_length
        self.max_length = max_length
        self.luke_model = self._load_luke_model(model_archive)

        self.mask_entity = mask_entity
        self.add_entity_token = add_entity_token

    def _load_luke_model(self, model_archive):
        entity_vocab = model_archive.entity_vocab
        bert_model_name = model_archive.bert_model_name
        model_config = model_archive.config
        model_weights = model_archive.state_dict

        model_config.vocab_size += 2
        word_emb = model_weights["embeddings.word_embeddings.weight"]
        head_emb = word_emb[self.tokenizer.convert_tokens_to_ids(["@"])[0]].unsqueeze(0)
        tail_emb = word_emb[self.tokenizer.convert_tokens_to_ids(["#"])[0]].unsqueeze(0)
        model_weights["embeddings.word_embeddings.weight"] = torch.cat(
            [word_emb, head_emb, tail_emb]
        )
        self.tokenizer.add_special_tokens(
            dict(additional_special_tokens=[HEAD_TOKEN, TAIL_TOKEN, MASK_TOKEN])
        )

        entity_emb = model_weights["entity_embeddings.entity_embeddings.weight"]
        mask_emb = entity_emb[entity_vocab[MASK_TOKEN]].unsqueeze(0).expand(2, -1)
        model_config.entity_vocab_size = 3
        model_weights["entity_embeddings.entity_embeddings.weight"] = torch.cat(
            [entity_emb[:1], mask_emb]
        )
        luke_model = LukeModel(model_config)
        luke_model.load_state_dict(model_weights, strict=False)

        return luke_model

    def forward(self, words, mask):
        word_segment_ids = torch.zeros(words.shape, dtype=torch.long)
        return self.luke_model(words, word_segment_ids, mask)[0]

    def tokenize(self, raw_tokens, pos_head, pos_tail):
        token_spans = dict(
            head=(pos_head[0], pos_head[-1] + 1), tail=(pos_tail[0], pos_tail[-1] + 1)
        )

        # Join the individul raw tokens together with spaces between them.
        # In the process, keep track of the characted indexes of the entities.
        text = ""
        cur = 0
        char_spans = dict(head=[None, None], tail=[None, None])
        for target_entity in ("head", "tail"):
            token_span = token_spans[target_entity]
            text += " ".join(raw_tokens[cur : token_span[0]])
            # if we're not at the begining, add a space
            if text:
                text += " "
            char_spans[target_entity][0] = len(text)
            text += " ".join(raw_tokens[token_span[0] : token_span[1]]) + " "
            char_spans[target_entity][1] = len(text)
            cur = token_span[1]
        text += " ".join(raw_tokens[cur:])
        text = text.rstrip()

        span_h, span_t = char_spans["head"], char_spans["tail"]

        tokens = [self.tokenizer.cls_token]

        # Insert special tokens before and after the entities.
        # if 'self.mask_entity' is true, also replaces the entity tokens with a
        # special mask token
        cur = 0
        token_spans = {}
        for span_name in ("span_h", "span_t"):
            span = eval(span_name)
            tokens += self.tokenizer.tokenize(text[cur : span[0]])
            start = len(tokens)
            if self.add_entity_token:
                tokens.append(HEAD_TOKEN if span_name == "span_h" else TAIL_TOKEN)
            entity_tokens = self.tokenizer.tokenize(text[span[0] : span[1]])
            if self.mask_entity:
                tokens += [MASK_TOKEN] * len(entity_tokens)
            else:
                tokens += entity_tokens
            if self.add_entity_token:
                tokens.append(HEAD_TOKEN if span_name == "span_h" else TAIL_TOKEN)
            token_spans[span_name] = (start, len(tokens))
            cur = span[1]

        tokens += self.tokenizer.tokenize(text[cur:])
        tokens.append(self.tokenizer.sep_token)

        # Convert tokens to ids
        word_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        entity_position_ids = []
        for span_name in ("span_h", "span_t"):
            span = token_spans[span_name]
            position_ids = list(range(span[0], span[1]))[: self.max_mention_length]
            position_ids += [-1] * (self.max_mention_length - span[1] + span[0])
            entity_position_ids.append(position_ids)

        entity_segment_ids = [0, 0]
        entity_attention_mask = [1, 1]

        # padding
        _pad_inputs(word_ids, self.max_length, 0)

        word_segment_ids = [0] * len(word_ids)
        word_attention_mask = np.zeros((self.max_length), dtype=np.int32)
        word_attention_mask[: len(tokens)] = 1

        return (
            word_ids,
            entity_position_ids[0][0],
            entity_position_ids[1][0],
            word_attention_mask,
        )


class ModularEncoder(nn.Module):
    """A modular class that combines an encoder and a pooler to go from input text to end representation."""
    def __init__(self, base_layer: nn.Module, base_layer_size: int, strategy: str):
        nn.Module.__init__(self)
        self.base = base_layer
        self.rep = create_rep_layer(base_layer_size, strategy)
    
    def forward(self, inputs):
        encoding = self.base(inputs["word"], inputs["mask"])
        representation = self.rep(encoding, inputs["pos1"], inputs["pos2"])
        return representation
    
    def tokenize(self, raw_tokens, pos_head, pos_tail):
        return self.base.tokenize(raw_tokens, pos_head, pos_tail)
