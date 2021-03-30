import sqlite3
from contextlib import closing
import json

connection = sqlite3.connect("image.db", check_same_thread=False)
connection.executescript(open("api/schema.sql").read())
connection.commit()

def add_paper(paper, i):
    query = """
    INSERT INTO Paper (PaperID, Title, Abstract, Year, AnnID) VALUES (?,?,?,?,?);
    """

    with closing(connection.cursor()) as c:
        c.execute(query, (paper["ids"], paper["title"], paper["abstract"], paper["year"], i))
        connection.commit()

def add_citation(paper):
    query = """
    INSERT INTO Citations (CitingPaperID, CitedPaperID) VALUES (?, ?);
    """

    citation_link = [(paper["ids"], cited_id) for cited_id in paper["pos"]]

    with closing(connection.cursor()) as c:
        c.executemany(query, citation_link)
        connection.commit()



with open("DBLP_train_test_dataset_1.json", "r") as f:
    data = json.load(f)

train = data["train"].values()

# for i, paper in enumerate(train):
#     add_paper(paper, i)

for paper in train:
    add_citation(paper)
