from flask import (
    Flask,
    request
    render_template
)

import torch
from transformers import AutoTokenizer
from Transformer.model.embedding_model import EmbeddingModel
from Transformer.candidate_selector.ann.ann_annoy import ANNAnnoy
# from Transformer.candidate_selector.ann.ann_candidate_selector import ANNCandidateSelector

import json
import numpy as np


class ANNCandidateSelector:
    def __init__(
        self, ann, neighbour_candidate, train_paper_dataset, mapping, extend_candidate=True
    ):
        self.ann = ann
        self.neighbour_candidate = neighbour_candidate
        self.extend_candidate = extend_candidate

        self.train_paper_dataset = train_paper_dataset
        self.mapping = mapping
        self.idx_paper_ids_mapping = list(mapping.keys())

    def get_candidate(self, query_embedding):
        candidate = self.ann.get_k_nearest_neighbour(
            query_embedding, self.neighbour_candidate
        )
        if self.extend_candidate:
            candidate_set = set(candidate)
            for i in candidate:
                citation_of_nn = self.train_paper_dataset[self.mapping[i]]["pos"]
                candidate_set.update([self.mapping[c] for c in citation_of_nn])

            candidate = list(candidate_set)

        similarity = self._get_similarity(query_embedding, candidate)

        result = [(self.idx_paper_ids_mapping[idx], sim) for idx, sim in zip(candidate, similarity)]
        result = sorted(result, key=lambda x: x[1])

        return result

    def _get_similarity(self, query_embedding, candidate_list):
        candidate_vector = np.array(self.ann.get_items_vector(candidate_list))
        sim = np.linalg.norm(candidate_vector - query_embedding, 2, axis=1)

        return sim.tolist()

app = Flask(__name__)

device = torch.device("cuda")

model = EmbeddingModel(config)
state_dict = torch.load(config.weight_path, map_location=device)["state_dict"]
model.load_state_dict(state_dict)
model = model.to(device)


tokenizer = AutoTokenizer.from_pretrained(config.model_name)

doc_embedding_vectors = torch.load(config.embedding_path)
doc_embedding_vectors = doc_embedding_vectors.cpu().numpy()
ann = ANNAnnoy.build_graph(doc_embedding_vectors)

with open("DBLP_train_test_dataset_1.json", "r") as f:
    dataset = json.load(f)["train"]

assert doc_embedding_vectors.shape[0] == len(dataset)

paper_ids_idx_mapping = {data["ids"]: idx for idx, data in enumerate(dataset)}

ann_candidate_selector = ANNCandidateSelector(
    ann, 8, dataset, paper_ids_idx_mapping
)

def to_device_dict(d, device):
    return {k: v.to(device) for k, v in d.items()}

@app.route('/', method="GET")
def index():
    return render_template("index.html")

@app.route('/recommend', method='POST')
def recommend():
    if request.method == 'POST':
        title = request.json['title']
        abstract = request.json['abstract']

        # TODO empty string for title and abstract

        encoded = tokenizer(
            title, 
            abstract,
            padding="max_length",
            max_length=256,
            truncation=True,
            return_tensors="pt"
        )

        encoded = to_device_dict(encoded, device)
        query_embedding = model(encoded)
        query_embedding_numpy = query_embedding.clone().cpu().numpy()[0]

        candidates = ann_candidate_selector.get_candidate(query_embedding_numpy)
    
        result = [{
            "title": dataset[c]["title"],
            "abstract": dataset[c]["abstract"]
        } for c in candidates]

        return render_template("recommend.html", result=result)

    # A database for all paper




