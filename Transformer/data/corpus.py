import datasets
import json
from tqdm import tqdm
import pickle


class S2ORCCorpus:
    def __init__(self, dataset, ids_idx_mapping):
        self.dataset = dataset
        self.ids_idx_mapping = ids_idx_mapping
    
    def __getitem__(self, key):
        return self.dataset[self.ids_idx_mapping[key]]

    def __iter__(self):
        return iter(self.ids_idx_mapping)

# train_ids = []
# with open("s2orc_train.json") as f:
#     for line in tqdm(f.readlines()):
#         data = json.loads(line)
#         train_ids.append(data["ids"])

# val_ids = []
# with open("s2orc_val.json") as f:
#     for line in tqdm(f.readlines()):
#         data = json.loads(line)
#         val_ids.append(data["ids"])

# mapping = {}
# for idx, ids in enumerate(tqdm(train_ids + val_ids)):
#     mapping[ids] = idx


# with open("train_ids.pkl", "wb") as g:
#     pickle.dump({
#         "train_ids": train_ids,
#         "val_ids": val_ids,
#         "paper_ids_idx_mapping": mapping
#     }, g
    # )

# dataset = S2ORCCorpus("train")
# dataset = datasets.load_dataset(
#     'json',
#     name="cs_paper",
#     data_files=["s2orc_train.json", "s2orc_val.json"],
#     split='train'
# )

# print(dataset[5000000])
# with open("train_ids.pkl", "rb") as g:
#     meta = pickle.load(g)

# print(meta["paper_ids_idx_mapping"][dataset[5000000]["ids"]])
