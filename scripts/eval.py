import pickle
import glob
import os
import json
import numpy as np

experiment = "results"
output_path = "./output/" + experiment + "/bert_base/"
P_path = glob.glob(output_path + "P*")
relations_collected = ["TREx", "GoogleRE", "Squad", "ConceptNet"]
relations = ["place_of_birth", "date_of_birth", "place_of_death"]
path_TREx_relations = "/mounts/work/kassner/LAMA/data/relations.jsonl"
relations_collected = ["GoogleRE"]
with open(path_TREx_relations, 'r') as json_file:
    json_list = list(json_file)

GoogleRE_relations = relations

TREx_relations = []
for json_str in json_list:
    result = json.loads(json_str)
    relations.append(result["relation"])
    TREx_relations.append(result["relation"])

#relations.append("Squad")
#relations.append("ConceptNet")

P = {}
results = {}
for r in relations:
    P_1 = 0
    P_1_BERT = 0
    P_1_NN = 0
    num = 0
    if os.path.isdir(output_path + r + "/"):
        with open(output_path + r + '/result.pkl', 'rb') as f:
            data = pickle.load(f)
        for d in data["list_of_results"]:
            P_1 += d['masked_topk']["P_AT_1"]
            P_1_BERT += d['masked_topk']["P_AT_1_BERT"]
            P_1_NN += d['masked_topk']["P_AT_1_NN"]
            num += 1
        results[r] = [P_1/num, P_1_BERT/num, P_1_NN/num]

overall = 0
num_overall = 0
"""correlations_BERT = {}
correlations_NN = {}
correlations_BERT_KNN = {}

for r in relations_collected:
    correlations_BERT[r] = []
    correlations_NN[r] = []
    correlations_BERT_KNN[r] = []"""

for r in relations_collected:
    if r=="TREx":
        num_P = 0
        P_collected = 0
        for P in TREx_relations:
            if P in results:
                P_collected += results[P]
                num_P += 1

        print(r)
        print(P_collected/num_P)
        num_overall += 1
        overall += P_collected/num_P
    elif r=="GoogleRE":
        num_P = 0
        P_collected = 0
        P_collected_BERT = 0
        P_collected_NN = 0

        for P in GoogleRE_relations:
            if P in results:
                P_collected += results[P][0]
                P_collected_BERT += results[P][1]
                P_collected_NN += results[P][2]
                num_P += 1

        print(r)
        print("BERT-kNN: ", P_collected/num_P)
        print("BERT: ", P_collected_BERT/num_P)
        print("kNN: ", P_collected_NN/num_P)

        num_overall += 1
        overall += P_collected/num_P

    else:
        print(r)
        print(results[r])
        num_overall += 1
        overall += results[r]

print(np.mean(list(results.values())))
