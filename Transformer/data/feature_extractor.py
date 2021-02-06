import json
from typing import List, Dict, Any
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


class FeatureExtractor:
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
    def get_hard_neg(
        query_paper_idx: int,
        paper_ids_idx_mapping: Dict[str, int],
        all_papers: List[Dict[str, Any]],
    ) -> List[str]:
        query_paper = all_papers[query_paper_idx]
        pos = set(query_paper["citations"])
        hard_neg = set()

        # Get postive of positive of query paper
        for p in pos:
            if p in paper_ids_idx_mapping:
                paper_idx = paper_ids_idx_mapping[p]
                hard_neg.update(all_papers[paper_idx]["citations"])
            else:
                logger.info(f"Abstract is not in paper with ids {p}")

        # Remove positive paper inside hard negative
        hard_neg = hard_neg - pos
        return list(hard_neg)


if __name__ == "__main__":
    feat = FeatureExtractor()
    path = "./dblp_dataset.json"
    papers = json.load(open(path, "r"))

    train_len = len(papers["train"])
    all_papers = papers["train"] + papers["test"]
    paper_ids_idx_mapping = feat.build_paper_ids_idx_mapping(all_papers)

    hard_neg_list = []
    for i in range(len(paper_ids_idx_mapping)):
        hard_neg_list.append(feat.get_hard_neg(i, paper_ids_idx_mapping, all_papers))

    assert len(all_papers) == len(hard_neg_list)

    for paper, hard_neg in zip(all_papers, hard_neg_list):
        paper["hard"] = hard_neg

    with open("./dblp_train_test_dataset.json", "w") as f:
        json.dump(
            {
                "mapping": paper_ids_idx_mapping,
                "train": all_papers[:train_len],
                "test": all_papers[train_len:],
            },
            f,
            indent=2,
        )
