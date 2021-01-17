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


def interpolate(distances, doc_scores, labels, predictions, topk=10):
    # normalizes NN probs
    normalized_distances = normalize_exp(distances[0])
    normalized_distances = [[normalized_distances]]

    probes_vocab_NN = torch.zeros(predictions.shape)
    unique_predictions = np.unique(labels)
    for p in unique_predictions:
        idcs_unique = np.argwhere(labels == p)
        probes_vocab_NN[p] = sum(normalized_distances[0][0][idcs_unique[0]])

    weighted = 0.3
    probes_combined = weighted*probes_vocab_NN + (1-weighted)*predictions

    max_probs, index_probs = torch.topk(input=probes_combined, k=topk, dim=0)
    max_probs_BERT, index_probs_BERT = torch.topk(input=predictions, k=topk,
                                                  dim=0)
    max_probs_NN, index_probs_NN = torch.topk(input=probes_vocab_NN, k=topk,
                                              dim=0)

    return (index_probs, max_probs, index_probs_BERT, max_probs_BERT,
            index_probs_NN, max_probs_NN, probes_combined, probes_vocab_NN)


def get_ranking_faster(predictions, log_probs, masked_indices, sample, vocab,
                       ranker, labels_dict_id, labels_dict, label_index=None,
                       index_list=None, P_AT=10, print_generation=True):
    P_AT_1 = 0.
    P_AT_1_NN = 0.0
    #P_AT_5 = 0.
    #P_AT_5_NN = 0.0
    #P_AT_10 = 0.
    #P_AT_10_NN = 0.0
    P_AT_1_BERT = 0.0
    #P_AT_5_BERT = 0.0
    #P_AT_10_BERT = 0.0
    vocab_r = list(vocab.keys())

    labels = []
    sentences = []
    label_tokens = []

    experiment_result = {}
    return_msg = ""

    #path_vectors = "./data/vectors/vectors_dump_"
    path_vectors = "/mounts/data/proj/kassner/BERT_kNN/data/vectors_drqa/vectors_dump_"
    path_vectors_retokenized1 = \
        "/mounts/work/kassner/BERT_kNN_oldcode/data/vectors_drqa_retokenized_hidden/vectors_dump_"
    path_vectors_retokenized2 = \
        "/mounts/data/proj/kassner/BERT_kNN/data/vectors_drqa_retokenized_hidden/vectors_dump_"


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
            try:
                xt = fvecs_read(path_vectors_retokenized1 + str(idcs[0]) + ".fvecs", count=count, offset=4*offset)
            except:
                xt = fvecs_read(path_vectors_retokenized2 + str(idcs[0]) + ".fvecs", count=count, offset=4*offset)

            #xt = fvecs_read(path_vectors + str(idcs[0]) +
            #                ".fvecs", count=count, offset=4*offset)
            xt = np.array(xt)
            index.add(xt)

            label_idcs = [(idcs[0], c+1) for c in range(idcs[2], len(xt)+idcs[2])]
            all_idcs.extend(label_idcs)
            scores = [score]*len(xt)
            doc_weights.extend(scores)

    doc_weights = np.array(doc_weights)
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
 
    doc_weights = doc_weights[top_k[0][0:idx_cut]]
    distances = [distances[0][0:idx_cut]]

    probs_combined, max_probs, probs_BERT, max_probs_BERT, probs_NN, max_probs_NN, probes_combined, probes_vocab_NN = \
        interpolate(distances, doc_weights, labels, log_probs)

    if label_index is not None:
        tokens = torch.from_numpy(np.asarray(label_index))
        lnprobs = torch.log(log_probs)
        label_perplexity = lnprobs.gather(dim=0, index=tokens,)
        lnprobs = torch.log(probes_combined)
        label_perplexity = lnprobs.gather(dim=0, index=tokens,)

        lnprobs = torch.log(probes_vocab_NN)
        label_perplexity = lnprobs.gather(dim=0, index=tokens,)

        # check if the labe_index should be converted to the vocab subset
        if index_list is not None:
            label_index = index_list.index(label_index)
        if len(labels) > 0:
            if label_index == probs_NN[0]:
                P_AT_1_NN = 1.
            """if label_index in probs_NN[0:5]:
                P_AT_5_NN = 1.
            if label_index in probs_NN[0:10]:
                P_AT_10_NN = 1."""
            if label_index == probs_combined[0]:
                P_AT_1 = 1.
            """if label_index in probs_combined[0:5]:
               P_AT_5 = 1.
            if label_index in probs_combined[0:10]:
                P_AT_10 = 1."""
            if label_index in probs_BERT[0]:
                P_AT_1_BERT = 1.
            """if label_index in probs_BERT[0:5]:
               P_AT_5_BERT = 1.
            if label_index in probs_BERT[0:10]:
                P_AT_10_BERT = 1."""
    probs_BERT = [vocab_r[idx] for idx in probs_BERT.tolist()]
    probs_combined = [vocab_r[idx] for idx in probs_combined.tolist()]
    probs_NN = [vocab_r[idx] for idx in probs_NN.tolist()]
    experiment_result["P_AT_1"] = P_AT_1
    """experiment_result["P_AT_5"] = P_AT_5
    experiment_result["P_AT_10"] = P_AT_10"""
    experiment_result["P_AT_1_NN"] = P_AT_1_NN
    """experiment_result["P_AT_5_NN"] = P_AT_5_NN
    experiment_result["P_AT_10_NN"] = P_AT_10_NN"""
    experiment_result["P_AT_1_BERT"] = P_AT_1_BERT
    """experiment_result["P_AT_5_BERT"] = P_AT_5_BERT
    experiment_result["P_AT_10_BERT"] = P_AT_10_BERT"""
    experiment_result["sentence"] = sentences
    experiment_result["topk_NN"] = probs_NN
    experiment_result["documents"] = list(doc_names)
    experiment_result["topk_BERT"] = probs_BERT
    experiment_result["topk_combined"] = probs_combined
    experiment_result["probs_NN"] = max_probs_NN.tolist()
    experiment_result["probs_BERT"] = max_probs_BERT.tolist()
    experiment_result["probes_combined"] = max_probs.tolist()
    experiment_result["document_scores"] = list(doc_scores)
    experiment_result["distances"] = distances
    experiment_result["labels"] = label_tokens

    experiment_result["sample"] = sample["masked_sentences"]
    return experiment_result, return_msg
