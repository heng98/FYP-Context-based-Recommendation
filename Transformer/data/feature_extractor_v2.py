import datasets
from tqdm import tqdm
import json
from collections import defaultdict
import random




dataset =  datasets.load_dataset(
    'json', 
    data_files={"train": "Dataset/s2orc/train_corpus.jsonl", "val": "Dataset/s2orc/val_corpus.jsonl"}
)

all_paper = []
with open("Dataset/s2orc/corpus.jsonl", "r") as f:
        for line in f.readlines():
            data = json.loads(line)
            if len(data["citations"]) >= 1:
                all_paper.append(data)

paper_ids_idx_mapping = {paper["ids"]: idx for idx, paper in enumerate(all_paper)}

def get_pos(paper):
    citations = paper.pop("citations")
    pos = [
        c for c in citations
        if c in paper_ids_idx_mapping
        and all_paper[paper_ids_idx_mapping[c]]["year"] <= paper["year"]
    ]
    if len(pos) > 30:
        pos = random.sample(pos, k=30)

    paper["pos"] = pos

    return paper

print("get pos")
dataset = dataset.map(get_pos, num_proc=2, writer_batch_size=100)
print(dataset["train"][0])