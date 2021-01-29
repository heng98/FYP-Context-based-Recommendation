import torch
from transformers import AutoTokenizer

from typing import NoReturn, List, Union, Dict, Any, Set
import json
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
# logger.setLevel(logging.INFO)
# console_handler = logging.StreamHandler()
# console_handler.setLevel(logging.INFO)
# logger.addHandler(console_handler)


class FeatureExtractor:
    def __init__(self, pretrained_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_path, use_fast=True)

    def get_input(
        self, title: Union[str, List[str]], abstract: Union[str, List[str]]
    ) -> Dict[str, torch.Tensor]:
        """Tokenization for all titles and abstracts

        Args:
        """
        data = self.tokenizer(
            title,
            abstract,
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors="pt",
        )
        return data

    @staticmethod
    def build_paper_ids_idx_mapping(papers: List[Dict[str, Any]]):
        mapping = dict()
        for idx, paper in enumerate(papers):
            mapping[paper["ids"]] = idx

        return mapping

    @staticmethod
    def get_pos(paper: Dict[str, Any]) -> Dict[str, Any]:
        pos = list(set(paper["citations"]))

        return pos

    @staticmethod
    def get_hard_neg(query_paper: str, network: Dict[str, Dict[str, Any]]) -> List[str]:
        pos = set(network[query_paper]["pos"])
        hard_neg = set()
        # Get postive of positive of query paper

        for p in pos:
            if p in network:
                hard_neg.update(network[p]["pos"])

            else:
                logger.info(f"Abstract is not in paper with ids {p}")

        # Remove positive paper inside hard negative
        hard_neg = hard_neg - pos

        return list(hard_neg)


if __name__ == "__main__":
    feat = FeatureExtractor("allenai/scibert_scivocab_cased")
    path = "./processed_aan_data/dataset.json"
    papers = json.load(open(path, "r"))

    all_papers = papers["train"] + papers["test"]
    titles = []
    abstracts = []
    for paper in all_papers:
        titles.append(paper["title"])
        abstracts.append(paper["abstract"])

    encoded = feat.get_input(titles, abstracts)
    paper_ids_idx_mapping = feat.build_paper_ids_idx_mapping(all_papers)

    torch.save(
        {"encoded": encoded.data, "paper_ids_idx_mapping": paper_ids_idx_mapping},
        "encoded.pth",
    )

    # Training Dataset
    train_network = defaultdict(dict)

    # Mapping of ids -> idx
    train_paper_ids_idx_mapping = {
        paper["ids"]: paper_ids_idx_mapping[paper["ids"]] for paper in papers["train"]
    }

    # Get all the positive from dataset
    for paper in papers["train"]:
        train_network[paper["ids"]]["pos"] = feat.get_pos(paper)

    # Get all the hard negative from network
    for p in train_network.keys():
        train_network[p]["hard"] = feat.get_hard_neg(p, train_network)

    torch.save(
        {
            "paper_ids_idx_mapping": train_paper_ids_idx_mapping,
            "network": train_network,
        },
        f"train_file.pth",
    )

    # Test Dataset
    test_network = defaultdict(dict)

    # Mapping of ids -> idx
    test_paper_ids_idx_mapping = {
        paper["ids"]: paper_ids_idx_mapping[paper["ids"]] for paper in papers["test"]
    }

    for paper in papers["test"]:
        test_network[paper["ids"]]["pos"] = feat.get_pos(paper)

    for p in test_network.keys():
        test_network[p]["hard"] = feat.get_hard_neg(p, test_network)

    torch.save(
        {
            "paper_ids_idx_mapping": test_paper_ids_idx_mapping,
            "network": test_network,
        },
        f"test_file.pth",
    )
