This repository contains code for ["BERT-kNN: Adding a kNN Search Component to Pretrained Language Models for Better QA"](https://arxiv.org/pdf/2005.00766.pdf).

## Citation
If you use this code, please cite:

    @misc{kassner2020bertknn,
          title={BERT-kNN: Adding a kNN Search Component to Pretrained Language Models for Better QA}, 
          author={Nora Kassner and Hinrich Schütze},
          year={2020},
          eprint={2005.00766},
          archivePrefix={arXiv},
          primaryClass={cs.CL}
    }

## Setup
### 1. Create conda environment and install requirements

(optional) It might be a good idea to use a separate conda environment. It can be created by running:
```
conda create -n bert_knn -y python=3.7 && conda activate bert_knn
pip install -r requirements.txt
```

export PYTHONPATH=${PYTHONPATH}:/path-to-project
### 2. Download the data
Download the LAMA data from:

```bash
wget https://dl.fbaipublicfiles.com/LAMA/data.zip
unzip data.zip
rm data.zip
```

## Pre-processing

### get Wikipedia corpus by DrQA
```bash
git clone https://github.com/facebookresearch/DrQA.git
```

### preprocess wikipedia
```bash
python batch_wikipdia.py --path_db_wirkipedia_drqa "DrAQ/data/wikipedia/docs.db"
```

### context embeddings
```bash
python encode_context.py --dump 0
```

### database target words
```bash
python db_target_creator.py
```

## Run BERT-kNN
```bash
python main.py
```

## Eval BERT-kNN
```bash
python eval_all.py
```
# BERT_kNN
