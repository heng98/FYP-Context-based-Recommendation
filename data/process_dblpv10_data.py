from paper_data_model import PaperDataModel

import json
from pathlib import Path

train = []
test = []
for file in Path("Dataset/dblp").glob("dblp-ref-*.json"):
    with open(file, "r") as f:
        for line in f:
            data = json.loads(line)

            if "abstract" in data:
                paper = PaperDataModel(
                    ids=data["id"],
                    title=data["title"],
                    abstract=data["abstract"],
                    citations=data["references"] if "references" in data else [],
                    year=data["year"],
                )

                if paper.year < 2014:
                    train.append(paper.to_dict())
                else:
                    test.append(paper.to_dict())

dataset = {
    "name": "DBLP",
    "train": train,
    'test': test
}

json.dump(dataset, open(f"dblp_dataset.json", 'w'), indent=2)