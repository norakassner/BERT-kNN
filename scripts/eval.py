import argparse
import json
import os
import numpy as np
import pickle

def main():
    path_TREx_relations = "./data/relations.jsonl"
    output_path = "./output/results/bert_base/"
    relations_GoogleRE = ["place_of_birth", "date_of_birth", "place_of_death"]
    relations_macro = {"TREx": [], "GoogleRE": relations_GoogleRE, "Squad": ["Squad"], "ConceptNet": ["ConceptNet"]}

    with open(path_TREx_relations, 'r') as json_file:
        json_list = list(json_file)

    for json_str in json_list:
        result = json.loads(json_str)
        relations_macro["TREx"].append(result["relation"])

    for r_macro in relations_macro:
        print(r_macro)
        P_1 = 0
        P_1_BERT = 0
        P_1_NN = 0
        num = 0
        for r_micro in relations_macro[r_macro]:
            if os.path.isdir(output_path + r_micro + "/"):
                with open(output_path + r_micro + '/result.pkl', 'rb') as f:
                    data = pickle.load(f)
                for d in data["list_of_results"]:
                    P_1 += d['masked_topk']["P_AT_1"]
                    P_1_BERT += d['masked_topk']["P_AT_1_BERT"]
                    P_1_NN += d['masked_topk']["P_AT_1_NN"]
                    num += 1
        print("BERT-kNN: ", P_1/num)
        print("BERT: ", P_1_BERT/num)
        print("kNN: ", P_1_NN/num)
        print("")

if __name__ == "__main__":
    main()
