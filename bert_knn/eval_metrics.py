import torch
import numpy as np
import faiss
import json


def normalize(distances, n=8):
    if sum(distances) != 0.0:
        distances = (1/np.power(distances, n)/sum(1/np.power(distances, n)))
    return distances


def normalize_exp(distances, n=6):
    if sum(distances) != 0.0:
        distances = (np.exp(-distances/n)/sum(np.exp(-distances/n)))
    return distances


def dump_jsonl(data, output_path, append=True):
    """
    Write list of objects to a JSON lines file.
    """
    mode = 'a+' if append else 'w'
    with open(output_path, mode, encoding='utf-8') as f:
        json_record = json.dumps(data, ensure_ascii=False)
        f.write(json_record + '\n')


def ivecs_write(fname, m):
    n, d = m.shape
    m1 = np.empty((n, d + 1), dtype='int32')
    m1[:, 0] = d
    m1[:, 1:] = m
    m1.tofile(fname)


def fvecs_write(fname, m):
    m = m.astype('float32')
    ivecs_write(fname, m.view('int32'))


def ivecs_read(fname, count=-1, offset=0):
    a = np.fromfile(fname, dtype='int32', count=count, offset=offset)
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def fvecs_read(fname, count=-1, offset=0):
    return ivecs_read(fname, count=count, offset=offset).view('float32')


def interpolate(distances, labels, predictions, topk=10):
    # normalizes NN probs
    normalized_distances = normalize_exp(distances[0])
    normalized_distances = [[normalized_distances]]

    probs_vocab_nn = torch.zeros(predictions.shape)
    unique_predictions = np.unique(labels)
    for p in unique_predictions:
        idcs_unique = np.argwhere(labels == p)
        probs_vocab_nn[p] = sum(normalized_distances[0][0][idcs_unique[0]])

    weighted = 0.3
    probs_combined = weighted*probs_vocab_nn + (1-weighted)*predictions

    probs_combined, vocab_idcs_combined = torch.topk(input=probs_combined, k=topk, dim=0)
    probs_bert, vocab_idcs_bert = torch.topk(input=predictions, k=topk,
                                                  dim=0)
    probs_nn, vocab_idcs_nn = torch.topk(input=probs_vocab_nn, k=topk,
                                              dim=0)

    return vocab_idcs_combined, probs_combined, vocab_idcs_bert, probs_bert, vocab_idcs_nn, probs_nn


def get_ranking(predictions, log_probs, sample, vocab, ranker, labels_dict_id, labels_dict, label_index=None,
                index_list=None):
    P_AT_1 = 0.
    P_AT_1_nn = 0.
    P_AT_1_bert = 0.

    vocab_r = list(vocab.keys())

    labels = []
    sentences = []
    label_tokens = []

    experiment_result = {}
    return_msg = ""

    path_vectors = "/mounts/work/kassner/BERT_kNN_oldcode/data/vectors_drqa_retokenized_hidden/vectors_dump_"

    N = 128
    num_ids = 3
    d = 768
    if "sub_label" in sample:
        if sample["sub_label"] == "squad":
            query = sample["masked_sentences"][0]
            query = query.replace("[MASK]", "")
            query = query.replace(".", "")
        else:
            query = sample["sub_label"]
    elif "sub" in sample:
        query = sample["sub"]

    doc_names, doc_scores = ranker.closest_docs(query, num_ids)
    filtered = [(name, score) for (name, score) in zip(doc_names, doc_scores)]

    if query.lower() in labels_dict_id and query.lower() not in doc_names:
        filtered = [(query.lower(), 1.0)]
    all_idcs = []
    doc_weights = []
    index = faiss.IndexFlatL2(d)

    for name, score in filtered:
        if name.lower() in labels_dict_id:
            idcs = labels_dict_id[name.lower()]
            count = ((idcs[3]+1)-idcs[2])+((idcs[3]+1)-idcs[2])*d
            offset = (d+1)*idcs[2]
            xt = fvecs_read(path_vectors + str(idcs[0]) + ".fvecs", count=count, offset=4*offset)
            xt = np.array(xt)
            index.add(xt)

            label_idcs = [(idcs[0], c+1) for c in range(idcs[2], len(xt)+idcs[2])]
            all_idcs.extend(label_idcs)
            scores = [score]*len(xt)
            doc_weights.extend(scores)

    # search for NN
    distances, top_k = index.search(np.array([predictions]), N)
    idx_cut = len(top_k[0])
    for idx, (k, d) in enumerate(zip(top_k[0], distances[0])):
        if k == -1:
            idx_cut = idx
            break
        else:

            label_idx = all_idcs[k]
            instance_id = "{:02}_{:08}".format(label_idx[0], label_idx[1])
            label_token = labels_dict.get_labels(instance_id).strip()
            label_vocab_idx = vocab[label_token]
            labels.append(int(label_vocab_idx))
            sentences.append(label_idx)
            label_tokens.append(label_token)

    distances = [distances[0][0:idx_cut]]

    vocab_idcs_combined, probs_combined, vocab_idcs_bert, probs_bert, vocab_idcs_nn, probs_nn = \
        interpolate(distances, labels, log_probs)

    if label_index is not None:

        # check if the labe_index should be converted to the vocab subset
        if index_list is not None:
            label_index = index_list.index(label_index)
        if len(labels) > 0:
            if label_index == vocab_idcs_nn[0]:
                P_AT_1_nn = 1.
            if label_index == vocab_idcs_combined[0]:
                P_AT_1 = 1.
            if label_index in vocab_idcs_bert[0]:
                P_AT_1_bert = 1.

    predictions_bert = [vocab_r[idx] for idx in vocab_idcs_bert.tolist()]
    predictions_combined = [vocab_r[idx] for idx in vocab_idcs_combined.tolist()]
    predictions_nn = [vocab_r[idx] for idx in vocab_idcs_nn.tolist()]
    experiment_result["P_AT_1"] = P_AT_1
    experiment_result["P_AT_1_nn"] = P_AT_1_nn
    experiment_result["P_AT_1_bert"] = P_AT_1_bert

    experiment_result["documents"] = list(doc_names)
    experiment_result["topk_bert"] = predictions_bert
    experiment_result["topk_combined"] = predictions_combined
    experiment_result["topk_nn"] = predictions_nn
    experiment_result["probs_nn"] = probs_nn.tolist()
    experiment_result["probs_bert"] = probs_bert.tolist()
    experiment_result["probs_combined"] = probs_combined.tolist()
    experiment_result["document_scores"] = list(doc_scores)
    experiment_result["labels"] = label_tokens

    experiment_result["sample"] = sample["masked_sentences"]
    return experiment_result, return_msg
