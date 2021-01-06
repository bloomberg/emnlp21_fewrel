import os
import json
import torch
from fewrel.fewshot_re_kit.sentence_encoder import CNNSentenceEncoder
from fewrel.fewshot_re_kit.modular_encoder import ModularEncoder, BERTBaseLayer, LukeBaseLayer, RobertaBaseLayer

def get_field(json_file, field_name):
    assert os.path.exists(json_file), "file not found: " + json_file
    with open(json_file, "r") as input:
        obj = json.load(input)
        assert field_name in obj, \
            "field '" + field_name + "' not found in json obj. Available fields are:\n\t" + str([str(key) for key in obj.keys()])
        return obj[field_name]

def reset_weights(model, ckpt):
    checkpoint = torch.load(ckpt, map_location=None if torch.cuda.is_available() else torch.device('cpu'))
    state_dict = checkpoint['state_dict']
    own_state = model.state_dict()
    for name, param in state_dict.items():
        if name.startswith("sentence_encoder.module."):
            name = name[len("sentence_encoder.module."):]
        if name.startswith("sentence_encoder."):
            name = name[len("sentence_encoder."):]

        if name not in own_state:
            raise Exception(
                "Unknown key in state dictionary: " + name +
                ". First own state is " + list(own_state.keys())[0])            
        own_state[name].copy_(param)

def get_encoder(encoder_name, 
        pretrain_ckpt, 
        max_length, 
        strategy, 
        load_ckpt=None, 
        mask_entity=False,
        no_entity_token=False):
    if encoder_name == 'cnn':
        prefix = pretrain_ckpt or 'glove/glove'
        try:
            mat_file = './pretrain/' + prefix + '_mat.npy'
            glove_mat = np.load(mat_file)
        except:
            raise Exception("Cannot find non-contextual embedding file: " + mat_file)
        try:
            index_file = './pretrain/'+ prefix + '_word2id.json'
            glove_word2id = json.load(open(index_file))
        except:
            raise Exception("Cannot find non-contextual embedding index file: " + index_file)
        
        sentence_encoder = CNNSentenceEncoder(
                glove_mat,
                glove_word2id,
                max_length)
    elif encoder_name == 'bert' or encoder_name == 'spanbert':
        pretrain_ckpt = pretrain_ckpt or './pretrain/bert-base-uncased'
        sentence_encoder = ModularEncoder(
            BERTBaseLayer(pretrain_ckpt, max_length, mask_entity, no_entity_token),
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
            LukeBaseLayer(pretrain_ckpt, max_length, mask_entity, no_entity_token),
            1024, # Hardcoded for now.
            strategy)
    else:
        raise NotImplementedError
    
    if load_ckpt:
        if encoder_name == 'cnn':
            raise Exception("Cannot load checkpoint for cnn model")
        reset_weights(sentence_encoder, load_ckpt)

    return sentence_encoder
