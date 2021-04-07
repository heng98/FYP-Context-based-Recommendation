import torch
from torch.utils.data import DataLoader

import numpy as np
from sklearn.metrics import ndcg_score
import datasets

from data.corpus import S2ORCCorpus

from model.reranker_model import SimpleReranker
from candidate_selector.ann.ann_annoy import ANNAnnoy
from candidate_selector.ann.ann_candidate_selector import ANNCandidateSelector
from ranker import Ranker, TransformerRanker
from evaluator import Evaluator

import argparse
import json
from tqdm import tqdm
import logging
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Collater:
    def __init__(self, tokenizer, max_seq_len):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __call__(self, batch):
        query_title = [data["query_paper"]["title"] for data in batch]
        query_abstract = [data["query_paper"]["abstract"] for data in batch]

        return self._encode(query_title, query_abstract)

    def _encode(self, title, abstract):
        return self.tokenizer(
            title,
            abstract,
            padding="max_length",
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_path", type=str)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--weight_path", type=str, required=True)
    parser.add_argument("--load_ann_path", type=str)
    parser.add_argument("--reranker_weight_path", type=str)
    config = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ranker_device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    document_embedding_model = AutoModel.from_pretrained(
        config.weight_path, return_dict=True, add_pooling_layer=False
    )

    # if config.reranker_weight_path:
    #     # reranker_model = AutoModelForSequenceClassification.from_pretrained(
    #     #     config.reranker_weight_path,
    #     #     return_dict=True,
    #     #     num_labels=1
    #     # )
    #     reranker_model = SimpleReranker()
    #     reranker_model = reranker_model.to(ranker_device)
    #     reranker_model.eval()

    tokenizer = AutoTokenizer.from_pretrained("allenai/cs_roberta_base")

    with open("Dataset/processed/s2orc_cs/train_ids.pkl", "rb") as g:
        meta = pickle.load(g)

    hf_dataset = datasets.load_dataset(
        "json",
        name="cs_paper",
        data_files=[
            "Dataset/processed/s2orc_cs/s2orc_train.json",
            "Dataset/processed/s2orc_cs/s2orc_val.json",
        ],
        split="train",
    )
    dataset = S2ORCCorpus(hf_dataset, meta["paper_ids_idx_mapping"])
    train_paper_ids = meta["train_ids"]
    val_paper_ids = meta["val_ids"]

    candidate_pool = set(train_paper_ids)

    # Embed papers in candidate pool
    if not config.load_ann_path:
        doc_embedding = np.empty((len(candidate_pool), 768), dtype="float32")
        dataloader = DataLoader(
            hf_dataset[: len(candidate_pool)],
            batch_size=8,
            collate_fn=Collater(tokenizer, 256),
            num_workers=8,
        )

        with torch.no_grad():
            for i, data in enumerate(dataloader):
                embedding = document_embedding_model(**data)["last_hidden_state"][:, 0]
                embedding = embedding.cpu().numpy()
                doc_embedding[i : i * 8] = embedding

        ann = ANNAnnoy.build(doc_embedding)
        ann.save("s2orc-cs.ann")

    else:
        ann = ANNAnnoy.load(config.load_ann_path)

    ann_candidate_selector = ANNCandidateSelector(
        ann,
        8,
        dataset,
        meta["paper_ids_idx_mapping"],
        candidate_pool
    )

    test_data = []
    with open("Dataset/processed/s2orc_cs/s2orc_test.json", "r") as f:
        for line in f:
            test_data.append(json.loads(line))

    evaluator = Evaluator(
        document_embedding_model,
        None,
        tokenizer,
        ann_candidate_selector,
        candidate_pool,
        device,
    )

    evaluator.evaluate(test_data)
    
    # title = "Context-based Recommendation"
    # abstract = "With the rapid growth of the scientific literature, citation recommendation systems able to speed up literature review and citing process during a research process. Recent approaches use bag-of-word retrieval to represent the documents, which discards word order information which is important in representation learning for documents. This project presents a method of recommend candidate references using document representations based on context of each document by learning document representations that incorporate inter-document document relatedness using citation graph and the state-of-the-art Transformer language model. Documents can be embedded into a high-dimensional vector space. Given a query document, it can be encoded into a vector which its nearest neighbours could be retrieved as candidates for citation. A recommendation web application is implemented to facilitate the citation recommendation."

    # recommendation = evaluator.recommend(title, abstract)
    # for p in recommendation:
    #     print(dataset[p[0]]["title"])