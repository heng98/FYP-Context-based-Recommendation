from flask import (
    Flask,
    request,
    render_template
)

import torch
from transformers import AutoTokenizer
from model.embedding_model import EmbeddingModel
from candidate_selector.ann.ann_annoy import ANNAnnoy
# from Transformer.candidate_selector.ann.ann_candidate_selector import ANNCandidateSelector

import json
import numpy as np

class Config:
    model_name = "allenai/scibert_scivocab_cased"
    max_seq_len = 256
    weight_path = "weights/dblp_with_nn_2/weights_1.pth"
    embedding_path = "embedding_dblp_2.pth"

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
                citation_of_nn = self.train_paper_dataset[self.idx_paper_ids_mapping[i]]["pos"]
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
config = Config()
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

paper_ids_idx_mapping = {ids: idx for idx, ids in enumerate(dataset.keys())}

ann_candidate_selector = ANNCandidateSelector(
    ann, 8, dataset, paper_ids_idx_mapping
)

def to_device_dict(d, device):
    return {k: v.to(device) for k, v in d.items()}

@app.route('/', methods=["GET"])
def index():
    return render_template("index.html")

@app.route('/recommend', methods=['POST'])
@torch.no_grad()
def recommend():
    if request.method == 'POST':
        title = request.form['title']
        abstract = request.form['abstract']

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
            "title": dataset[c[0]]["title"],
            "abstract": dataset[c[0]]["abstract"]
        } for c in candidates]

        return render_template("recommend.html", result=result)

    # A database for all paper

if __name__ == '__main__':
    app.run(debug=True)




