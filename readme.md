# Realistic Few-Shot Relation Extraction
This repository contains code to reproduce the results in the paper "Towards Realistic Few-Shot Relation Extraction" to appear in The 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP 2021). This code is not intended to be modified or reused. It is a fork of an existing [FewRel repository](https://github.com/thunlp/FewRel) with some modifications.

## Fine-tuning

The following command is to fine-tune a pre-trained model on a training dataset complying with the FewRel's format (see the [Dataset section](#dataset) below).

```bash
python -m fewrel.fewrel_eval \
  --train train_wiki \
  --test val_wiki \
  --encoder {"cnn", "bert", "roberta", "luke"} \
  --pool {"cls", "cat_entity_reps"} \
  --data_root data/fewrel \
  --pretrain_ckpt {pretrained_model_path} \
  --train_iter 10000 \
  --val_iter 1000 \
  --val_step 2000 \
  --test_iter 2000
```

The above command will dump the fine-tuned model under `./checkpoint`. The following command can be used to get the overall accuracy for the fine-tuned model. 

## Overall accuracy
```bash
python -m fewrel.fewrel_eval \
  --only_test \
  --test val_wiki \
  --encoder {"cnn", "bert", "roberta", "luke"} \
  --pool {"cls", "cat_entity_reps"} \
  --data_root data/fewrel \
  --pretrain_ckpt {pretrained_model_path} \ # needed for getting model config
  --load_ckpt {trained_checkpoint_path} \
  --test_iter 2000
```

## P@50 for individual relations

Precision at 50 can be calculated using the following command

```bash
python -m fewrel.alt_eval \
  --test {test_file_name_without_extension} \ # e.g., tacred_org 
  --encoder {"cnn", "bert", "roberta", "luke"} \
  --pool {"cls", "cat_entity_reps"} \
  --data_root {path_to_data_folder} \
  --pretrain_ckpt {pretrained_model_path} \ # needed for getting model config
  --load_ckpt {trained_checkpoint_path}
```

## Pre-trained models

In this work, several encoders are experimented with including _CNN_, _BERT_, _SpanBERT_, _RoBERTa-base_, _RoBERTa-large_, and _LUKE-base_. Most pre-trained models can be downloaded from [Hugging Face Transformers](https://huggingface.co/transformers/pretrained_models.html), and _LUKE-base_ can be downloaded from its original [GitHub repository](https://github.com/studio-ousia/luke).

**Note:** the original LUKE code depends on an older version of HuggingFace Transformers, which is not compatible with the version used in this repository. To experiment with LUKE, please run script `./checkout_out_luke.sh`. This will first clone the original LUKE repository, apply the necessary changes to make luke compatible with this repo, and move the LUKE module to the correct place to make sure the code runs correctly.

## Dataset

The original FewRel dataset has already be contained in the github repo (here)[./data/fewrel]. To convert other dataset (e.g., [TACRED](https://nlp.stanford.edu/projects/tacred/)) to the FewRel format, one could use [./scripts/prep_more_data.py](./scripts/prep_more_data.py).

[./scripts/select_rel.py](./scripts/select_rel.py) is a script to augment an existing dataset with relations from another dataset. For example, to add a list of relations from dataset `source.json` to `destination.json` and dump the merged dataset to a file `output.json`, one can use the following command:

```bash
python scripts/select_rel.py add_rel \
  --src source.json \
  --dst destination.json \
  --output output.json \
  --rels {relations_delimitated_by_space}
```
