import sqlite3
from contextlib import closing
import json
import pickle
import datasets
from tqdm import tqdm

connection = sqlite3.connect("s2orc.db", check_same_thread=False)
connection.executescript(open("api/schema.sql").read())
connection.commit()

def add_paper(paper, i):
    query = """
    INSERT INTO Paper (PaperID, Title, Abstract, Year, AnnID) VALUES (?,?,?,?,?);
    """

    with closing(connection.cursor()) as c:
        c.execute(query, (paper["ids"], paper["title"], paper["abstract"], paper["year"], i))
        connection.commit()



hf_dataset = datasets.load_dataset(
    'json',
    name="cs_paper",
    data_files=["Dataset/processed/s2orc_cs/s2orc_train.json", "Dataset/processed/s2orc_cs/s2orc_val.json"],
    split='train'
)


# query = """
# INSERT INTO Paper (PaperID, Title, Abstract, Year, AnnID) VALUES (?,?,?,?,?);
# """
# with closing(connection.cursor()) as c:
#     for i, paper in enumerate(tqdm(hf_dataset)):
#         c.execute(query, (paper["ids"], paper["title"], paper["abstract"], paper["year"], i))
#         if (i + 1) % 200000 == 0:
#             connection.commit()

#     connection.commit()


with open("Dataset/processed/s2orc_cs/train_ids.pkl", "rb") as g:
    meta = pickle.load(g)

candidate = set(meta["train_ids"] + meta["val_ids"])

def add_citation(paper):
    query = """
    INSERT INTO Citations (CitingPaperID, CitedPaperID) VALUES (?, ?);
    """

    citation_link = [(paper["ids"], cited_id) for cited_id in (set(paper["pos"]) & candidate)]

    with closing(connection.cursor()) as c:
        c.executemany(query, citation_link)
        connection.commit()

for paper in tqdm(hf_dataset):
    add_citation(paper)
