import argparse
from bert_knn.modules import build_model_by_name
import numpy as np
import os
from tqdm import tqdm


def ivecs_write(fname, m):
    n, d = m.shape
    m1 = np.empty((n, d + 1), dtype='int32')
    m1[:, 0] = d
    m1[:, 1:] = m
    m1.tofile(fname)


def fvecs_write(fname, m):
    m = m.astype('float32')
    ivecs_write(fname, m.view('int32'))


def main(num_dump):
    input_param = {
        "lm": "bert",
        "label": "bert_base",
        "models_names": ["bert"],
        "bert_model_name": "bert-base-uncased",
    }
    args = argparse.Namespace(**input_param)
    model = build_model_by_name(input_param["lm"], args)

    print("DUMP: ", num_dump)

    data_file = "./data/wikidump_batched/dump_"
    save_dir = "./data/vectors/"
    save_file = save_dir + "vectors_dump_"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    if os.path.isfile(save_dir + str(num_dump) + ".fvecs"):
        raise RuntimeError('%s already exists! Not overwriting.' % save_dir)
    num_sentences = 0
    with open(data_file + str(num_dump) + "_sentences.txt") as f_data:
        for line in f_data:
            line = line.strip()
            num_sentences += 1

    sentence_batch = []
    batch_size = 100
    num_batches = 0
    with open(save_file + str(num_dump) + ".fvecs", "wb") as f_out:
        with open(data_file + str(num_dump) + "_sentences.txt", "r") as f_in:
            for line in tqdm(f_in, total=num_sentences):
                line = line.strip()
                sentence_batch.append([line])
                if len(sentence_batch) == batch_size:
                    out = model.get_hidden_state(sentence_batch)
                    fvecs_write(f_out, np.array(out))
                    sentence_batch = []
                    num_batches += 1

            if len(sentence_batch) != 0:
                out = model.get_hidden_state(sentence_batch)
                fvecs_write(f_out, np.array(out))
                sentence_batch = []


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump",
                        default=0,
                        type=int,
                        required=True,
                        help="Which dump should be processed")
    args = parser.parse_args()
    main(args.dump)
