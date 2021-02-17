from paper_data_model import PaperDataModel

import json
from pathlib import Path

papers = []
with open("Dataset/dblp/corpus.json", "r") as f:
    for line in f:
        data = json.loads(line)
        # print(data.keys())
        if ("abstract" in data) and ("title" in data):
            paper = PaperDataModel(
                ids=data["id"],
                title=data["title_raw"],
                abstract=data["abstract_raw"],
                citations=data["out_citations"],
                year=data["year"],
            )

            papers.append(paper.to_dict())


dataset = {
    "name": "DBLP",
    "papers": papers
}

json.dump(dataset, open(f"dblp_dataset.json", 'w'), indent=2)
