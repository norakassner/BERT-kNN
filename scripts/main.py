# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from interpolated import main as run_evaluation
from interpolated import load_file
from bert_knn.modules import build_model_by_name

import argparse
import pprint
import statistics
from os import listdir
import os
from os.path import isfile, join
from shutil import copyfile
from collections import defaultdict
import numpy as np
import time
import json
from drqa import retriever
import sqlite3


class LabelDB(object):
    """Sqlite backed document storage.
    Implements get_doc_text(doc_id).
    """

    def __init__(self, db_path=''):
        self.path = db_path
        self.connection = sqlite3.connect(self.path, check_same_thread=False)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def path(self):
        """Return the path to the file that backs this database."""
        return self.path

    def close(self):
        """Close the connection to the database."""
        self.connection.close()

    def get_instance_id(self):
        """Fetch all ids of docs stored in the db."""
        cursor = self.connection.cursor()
        cursor.execute("SELECT instance_id FROM labels")
        results = [r[0] for r in cursor.fetchall()]
        cursor.close()
        return results

    def get_labels(self, instance_id):
        """Fetch the raw text of the doc for 'doc_id'."""
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT label FROM labels WHERE instance_id = ?",
            (instance_id,)
        )
        result = cursor.fetchone()
        cursor.close()
        return result if result is None else result[0]

LMs = [
    {
        "lm":
        "bert",
        "label":
        "bert_base",
        "models_names": ["bert"],
        "bert_model_name":
        "bert-base-uncased",
    }

]


def run_experiments(
    relations,
    data_path_pre,
    data_path_post,
    input_param,
    index_faiss=None,
    labels_dict_id=None,
    labels_dict=None,
):
    model = None
    pp = pprint.PrettyPrinter(width=41, compact=True)

    all_Precision1 = []
    type_Precision1 = defaultdict(list)
    type_count = defaultdict(list)

    results_file = open("last_results.csv", "w+")

    for relation in relations:
        pp.pprint(relation)
        PARAMETERS = {
            "dataset_filename": "{}{}{}".format(
                data_path_pre, relation["relation"], data_path_post
            ),
            "template": "",
            "batch_size": 224,
            "logdir": "output",
            "full_logdir": "output/results/{}/{}".format(
                input_param["label"], relation["relation"]
            ),
            "lowercase": True,
            "max_sentence_length": 100,
            "threads": -1,
            "interactive": False,
        }

        if "template" in relation:
            PARAMETERS["template"] = relation["template"]

        PARAMETERS.update(input_param)

        args = argparse.Namespace(**PARAMETERS)

        # see if file exists
        try:
            data = load_file(args.dataset_filename)
        except Exception as e:
            print("Relation {} excluded.".format(relation["relation"]))
            print("Exception: {}".format(e))
            continue

        if model is None:
            [model_type_name] = args.models_names
            model = build_model_by_name(model_type_name, args)

        Precision1 = run_evaluation(args, ranker=index_faiss,
                                    labels_dict_id=labels_dict_id,
                                    labels_dict=labels_dict, shuffle_data=False,
                                    model=model)
        print("P@1 : {}".format(Precision1), flush=True)
        all_Precision1.append(Precision1)

        results_file.write(
            "{},{}\n".format(relation["relation"], round(Precision1 * 100, 2))
        )
        results_file.flush()

        if "type" in relation:
            type_Precision1[relation["type"]].append(Precision1)
            data = load_file(PARAMETERS["dataset_filename"])
            type_count[relation["type"]].append(len(data))

    mean_p1 = statistics.mean(all_Precision1)
    print("@@@ {} - mean P@1: {}".format(input_param["label"], mean_p1))
    results_file.close()

    for t, l in type_Precision1.items():

        print(
            "@@@ ",
            input_param["label"],
            t,
            statistics.mean(l),
            sum(type_count[t]),
            len(type_count[t]),
            flush=True,
        )

    return mean_p1, all_Precision1


def get_TREx_parameters(data_path_pre="/mounts/work/kassner/LAMA/data/"):
    relations = load_file("{}relations.jsonl".format(data_path_pre))
    data_path_pre += "TREx/"
    data_path_post = ".jsonl"
    return relations, data_path_pre, data_path_post


def get_GoogleRE_parameters():
    relations = [
        {"relation": "place_of_birth", "template": "[X] was born in [Y] ."},
        {"relation": "date_of_birth", "template": "[X] (born [Y])."},
        {"relation": "place_of_death", "template": "[X] died in [Y] ."}]
    data_path_pre = "/mounts/work/kassner/LAMA/data/Google_RE/"
    data_path_post = "_test.jsonl"
    return relations, data_path_pre, data_path_post


def get_ConceptNet_parameters(data_path_pre="/mounts/work/kassner/LAMA/data/"):
    relations = [{"relation": "ConceptNet"}]
    data_path_pre += "ConceptNet/"
    data_path_post = ".jsonl"
    return relations, data_path_pre, data_path_post


def get_Squad_parameters(data_path_pre="/mounts/work/kassner/LAMA/data/"):
    relations = [{"relation": "Squad"}]
    data_path_pre += "Squad/"
    data_path_post = ".jsonl"
    return relations, data_path_pre, data_path_post


def run_all_LMs(parameters, index_faiss=None, labels_dict_id=None,
                labels_dict=None):
    for ip in LMs:
        print(ip["label"])
        run_experiments(*parameters, input_param=ip, index_faiss=index_faiss,
                        labels_dict_id=labels_dict_id,
                        labels_dict=labels_dict)


if __name__ == "__main__":

    # load label dict
    labels_dict_path = "./data/wikidump_batched/dict_id_idcs.json"
    with open(labels_dict_path, 'r') as f:
        labels_dict_id = json.load(f)

    ranker = retriever.get_class('tfidf')(tfidf_path=None)
    database_path = "./data/labels.db"
    labels_dict = LabelDB(database_path)

    print("1. Google-RE")
    parameters = get_GoogleRE_parameters()
    run_all_LMs(parameters, index_faiss=ranker, labels_dict_id=labels_dict_id,
                labels_dict=labels_dict)

    print("3. ConceptNet")
    parameters = get_ConceptNet_parameters()
    run_all_LMs(parameters, index_faiss=ranker, labels_dict_id=labels_dict_id,
                labels_dict=labels_dict)

    """print("2. T-REx")
    parameters = get_TREx_parameters()
    run_all_LMs(parameters, index_faiss=ranker, labels_dict_id=labels_dict_id,
                labels_dict=labels_dict)"""

    """print("4. SQuAD")
    parameters = get_Squad_parameters()
    run_all_LMs(parameters, index_faiss=ranker, labels_dict_id=labels_dict_id,
                labels_dict=labels_dict)"""
