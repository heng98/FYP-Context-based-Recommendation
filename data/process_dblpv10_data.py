from paper_data_model import PaperDataModel

import json
from pathlib import Path

papers = []
for file in Path("Dataset/dblp").glob("dblp-ref-*.json"):
    with open(file, "r") as f:
        for line in f:
            data = json.loads(line)

            if ("abstract" in data) and ("title" in data):
                paper = PaperDataModel(
                    ids=data["id"],
                    title=data["title"],
                    abstract=data["abstract"],
                    citations=data["references"] if "references" in data else [],
                    year=data["year"],
                )

                papers.append(paper.to_dict())


dataset = {
    "name": "DBLP",
    "papers": papers
}

json.dump(dataset, open(f"dblp_dataset.json", 'w'), indent=2)