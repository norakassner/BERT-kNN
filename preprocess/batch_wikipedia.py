import os
import sqlite3
import unicodedata
import transformers
import json
from nltk.corpus import stopwords
import string
import argparse


class DocDB(object):
    """Sqlite backed document storage.
    Implements get_doc_text(doc_id).
    """

    def __init__(self, db_path):
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

    def get_doc_ids(self):
        """Fetch all ids of docs stored in the db."""
        cursor = self.connection.cursor()
        cursor.execute("SELECT id FROM documents")
        results = [r[0] for r in cursor.fetchall()]
        cursor.close()
        return results

    def get_doc_text(self, doc_id):
        """Fetch the raw text of the doc for 'doc_id'."""
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT text FROM documents WHERE id = ?",
            (normalize(doc_id),)
        )
        result = cursor.fetchone()
        cursor.close()
        return result if result is None else result[0]


def dump_jsonl(data, output_path, append=True):
    """
    Write list of objects to a JSON lines file.
    """
    mode = 'a+' if append else 'w'
    with open(output_path, mode, encoding='utf-8') as f:
        json_record = json.dumps(data, ensure_ascii=False)
        f.write(json_record + '\n')


def normalize(text):
    """Resolve different type of unicode encodings."""
    return unicodedata.normalize('NFD', text)


def main(path_drqa):
    num_dumps = 100

    # wikipedia data base
    path_db = path_drqa
    db = DocDB(path_db)
    db_ids = db.get_doc_ids()
    ids_per_dump = int(len(db_ids)/num_dumps) + 1

    save_dir = "./data/wikidump_batched/"
    save_file = save_dir + "dump_"

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    save_files_sentences = {}
    save_files_labels = {}
    save_files_dbids = {}
    for n in range(num_dumps):
        if not os.path.exists(save_dir + str(n) + "_dbids.txt"):
            save_files_dbids[n] = open(save_file + str(n) + "_dbids.txt", "w")
            save_files_sentences[n] = open(save_file + str(n) + "_sentences.txt", "w")
            save_files_labels[n] = open(save_file + str(n) + "_labels.txt", "w")

    tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
    stop_words = set(stopwords.words('english'))

    save_idx = 0
    num_lines = 0
    print("start masking")
    for id in db_ids:
        raw_text = db.get_doc_text(id)
        raw_list = raw_text.split("\n")
        raw_list = list(filter(None, raw_list))
        num_lines += 1
        for line in raw_list:
            sentences = list(filter(None, line.split(".")))
            for sentence in sentences:
                sentence = sentence.strip()
                tokens = tokenizer.tokenize((sentence))
                for idx, token in enumerate(tokens):
                    if token in tokenizer.vocab and token[0] != "#" and token not in stop_words and token not in string.punctuation:
                        masked_sentences = \
                            (" ".join(tokens[:idx]) + " [MASK] " + " ".join(tokens[idx+1:]) + ".")
                        save_files_sentences[save_idx].write(masked_sentences)
                        save_files_sentences[save_idx].write("\n")
                        save_files_labels[save_idx].write(token)
                        save_files_labels[save_idx].write("\n")
                        save_files_dbids[save_idx].write(id)
                        save_files_dbids[save_idx].write("\n")

            if num_lines > ids_per_dump:
                print(save_idx, num_dumps)
                num_lines = 0
                save_files_sentences[save_idx].close()
                save_files_labels[save_idx].close()
                save_files_dbids[save_idx].close()
                save_idx += 1
    save_files_sentences[save_idx].close()
    save_files_labels[save_idx].close()
    save_files_dbids[save_idx].close()

    id_dict = {}
    for d in range(num_dumps):
        with open(save_file + str(d) + "_dbids.txt") as f:
            idx = 0
            num_ent = 0
            for line in f:
                line = line.strip()
                if line not in id_dict:
                    id_dict[line.lower()] = [d, num_ent, idx, idx]
                    num_ent += 1
                else:
                    id_dict[line][3] += 1
                idx += 1

    with open(save_file + "dict_id_idcs.json", "w") as f:
        json.dump(id_dict, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_db_wirkipedia_drqa",
                        default=0,
                        type=str,
                        required=True,
                        help="Path_drqa")
    args = parser.parse_args()
    main(args.path_db_wirkipedia_drqa)
