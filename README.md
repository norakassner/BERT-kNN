This repository contains code for ["BERT-kNN: Adding a kNN Search Component to Pretrained Language Models for Better QA"](https://arxiv.org/pdf/2005.00766.pdf).
The repository is forked from https://github.com/facebookresearch/LAMA and adapted accordingly.

## Citation
If you use this code, please cite:
```bibtex
@inproceedings{kassner-schutze-2020-bert,
    title = "{BERT}-k{NN}: Adding a k{NN} Search Component to Pretrained Language Models for Better {QA}",
    author = {Kassner, Nora  and
      Sch{\"u}tze, Hinrich},
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2020",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.findings-emnlp.307",
    doi = "10.18653/v1/2020.findings-emnlp.307",
    pages = "3424--3430",
    abstract = "Khandelwal et al. (2020) use a k-nearest-neighbor (kNN) component to improve language model performance. We show that this idea is beneficial for open-domain question answering (QA). To improve the recall of facts encountered during training, we combine BERT (Devlin et al., 2019) with a traditional information retrieval step (IR) and a kNN search over a large datastore of an embedded text collection. Our contributions are as follows: i) BERT-kNN outperforms BERT on cloze-style QA by large margins without any further training. ii) We show that BERT often identifies the correct response category (e.g., US city), but only kNN recovers the factually correct answer (e.g.,{``}Miami{''}). iii) Compared to BERT, BERT-kNN excels for rare facts. iv) BERT-kNN can easily handle facts not covered by BERT{'}s training set, e.g., recent events.",
}

@inproceedings{petroni2019language,
  title={Language Models as Knowledge Bases?},
  author={F. Petroni, T. Rockt{\"{a}}schel, A. H. Miller, P. Lewis, A. Bakhtin, Y. Wu and S. Riedel},
  booktitle={In: Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP), 2019},
  year={2019}
}
```
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
python scripts/main.py
```

## Eval BERT-kNN
```bash
python scripts/eval_all.py
```
