# FewRel Dataset, Toolkits and Baseline Models

**Update: We have released FewRel 2.0! The paper, dataset and codes for FewRel 2.0 have already been uploaded to this repo. The evaluation website and instructions will be updated in a few days.**

FewRel is a large-scale few-shot relation extraction dataset, which contains 70000 natural language sentences expressing 100 different relations. This dataset is presented in the our EMNLP 2018 paper [FewRel: A Large-Scale Few-Shot Relation Classification Dataset with State-of-the-Art Evaluation](https://github.com/thunlp/FewRel/blob/master/paper/fewrel.pdf).

More info at https://thunlp.github.io/fewrel.html .

## Citing
If you used our data, toolkits or baseline models, please kindly cite our paper:
```
@inproceedings{han2018fewrel,
               title={FewRel:A Large-Scale Supervised Few-Shot Relation Classification Dataset with State-of-the-Art Evaluation},
               author={Han, Xu and Zhu, Hao and Yu, Pengfei and Wang, Ziyun and Yao, Yuan and Liu, Zhiyuan and Sun, Maosong},
               booktitle={EMNLP},
               year={2018}
}
```


If you have questions about any part of the paper, submission, leaderboard, codes, data, please e-mail zhuhao15@mails.tsinghua.edu.cn.

## Contributions

Hao Zhu first proposed this problem and proposed the way to build the dataset and the baseline system; Ziyuan Wang built and maintained the crowdsourcing website; Yuan Yao helped download the original data and conducted preprocess; 
Xu Han, Hao Zhu, Pengfei Yu and Ziyun Wang implemented baselines and wrote the paper together; Zhiyuan Liu provided thoughtful advice and funds through the whole project. The order of the first four authors are determined by dice rolling. 

## Dataset and Pretrain files

The dataset has already be contained in the github repo. However, due to the large size, glove files (pre-trained word embeddings) and BERT pretrain checkpoint are not included. Please download `pretrain.tar` from [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/58f57bda00eb40be8d10/?dl=1) and put it under the root. Then run `tar xvf pretrain.tar` to decompress it.

## Usage

To run our baseline models, use command

```bash
python train_demo.py {MODEL_NAME}
```

replace `{MODEL_NAME}` with `proto`, `metanet`, `gnn` or `snail`.


