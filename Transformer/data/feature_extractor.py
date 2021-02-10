import json
from typing import List, Dict, Any
import json
import logging

import argparse
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

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
        pos = set(query_paper["pos"])
        hard_neg = set()

        # Get postive of positive of query paper
        for p in pos:
            if p in paper_ids_idx_mapping:
                paper_idx = paper_ids_idx_mapping[p]
                hard_neg.update(all_papers[paper_idx]["pos"])
            else:
                logger.info(f"Abstract is not in paper with ids {p}")

        # Remove positive paper inside hard negative
        hard_neg = hard_neg - pos
        return list(hard_neg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--min_citation", type=int, default=-1)
    parser.add_argument("--min_year", type=int, default=-1)

    parser.add_argument("--split_year", type=int, required=True)
    parser.add_argument("--save_dir", type=str, required=True)

    args = parser.parse_args()

    feat = FeatureExtractor()
    papers = json.load(open(args.data_path, "r"))

    dataset_name = papers["name"]

    # candidate_paper = papers["papers"]
    candidate_paper = list(
        filter(
            lambda x: (x["year"] >= args.min_year)
            and (len(x["citations"]) >= args.min_citation),
            papers["papers"],
        )
    )

    paper_ids_idx_mapping = feat.build_paper_ids_idx_mapping(candidate_paper)

    def get_pos(p):
        p["pos"] = list(set(p.pop("citations")) & set(paper_ids_idx_mapping))
        return p

    def get_hard_neg(i):
        hard_neg = feat.get_hard_neg(i, paper_ids_idx_mapping, candidate_paper)
        candidate_paper[i]["hard_neg"] = hard_neg

        return candidate_paper[i]

    with ProcessPoolExecutor() as executor:
        candidate_paper = list(tqdm(executor.map(get_pos, candidate_paper)))

    logger.info("Extracting Hard Neg")
    with ProcessPoolExecutor() as executor:
        candidate_paper = list(
            tqdm(executor.map(get_hard_neg, range(len(candidate_paper))))
        )

    train_paper = list(
        filter(lambda x: (x["year"] <= args.split_year), candidate_paper)
    )
    test_paper = list(filter(lambda x: (x["year"] > args.split_year), candidate_paper))

    with open(f"./{dataset_name}_train_test_dataset.json", "w") as f:
        json.dump(
            {
                "train": train_paper,
                "test": test_paper,
            },
            f,
            indent=2,
        )
