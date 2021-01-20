import os
import sqlite3


def create(db_file):
    if os.path.isfile(db_file):
        raise RuntimeError('%s already exists! Not overwriting.' % db_file)
    # conncetion
    conn = sqlite3.connect(db_file)

    c = conn.cursor()
    c.execute("CREATE TABLE labels (instance_id char(11) PRIMARY KEY, label text) WITHOUT ROWID;")

    return conn, c


def main():
    data_dir = "./data/wikidump_batched/dump_"
    db_file = "./data/labels.db"

    conn, c = create(db_file)
    for dump in range(100):
        data = []
        with open(data_dir + str(dump) + "_labels.txt") as f:
            for idx, line in enumerate(f):
                label = line.strip()
                instance_id = "{:02}_{:08}".format(dump, int(idx))
                print(instance_id)
                data.append([instance_id, label])
        c.executemany("INSERT INTO labels VALUES (?,?)", data)
        conn.commit()
    conn.close()


if __name__ == '__main__':
    main()
