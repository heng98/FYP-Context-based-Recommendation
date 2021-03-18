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

    parser.add_argument("--train_range", type=str, required=True)
    parser.add_argument("--val_range", type=str, required=True)
    parser.add_argument("--test_range", type=str, required=True)

    parser.add_argument("--save_dir", type=str, required=True)

    args = parser.parse_args()

    feat = FeatureExtractor()
    papers = json.load(open(args.data_path, "r"))

    train_range = tuple(int(year) for year in args.train_range.split("_"))
    val_range = tuple(int(year) for year in args.val_range.split("_"))
    test_range = tuple(int(year) for year in args.test_range.split("_"))

    logger.info(f"Training range: From {train_range[0]} to {train_range[1]} inclusive")
    logger.info(f"Validation range: From {val_range[0]} to {val_range[1]} inclusive")
    logger.info(f"Testing range: From {test_range[0]} to {test_range[1]} inclusive")

    dataset_name = papers["name"]

    all_paper = papers["papers"]
    all_paper = list(
        filter(
            lambda x: (x["year"] >= args.min_year)
            and (len(x["citations"]) >= args.min_citation),
            papers["papers"],
        )
    )

    paper_ids_idx_mapping = feat.build_paper_ids_idx_mapping(all_paper)

    def get_pos(p):
        citations = p.pop("citations")
        pos = [
            c
            for c in citations
            if c in paper_ids_idx_mapping
            and all_paper[paper_ids_idx_mapping[c]]["year"] <= p["year"]
        ]
        p["pos"] = pos
        return p

    def get_hard_neg(i):
        hard_neg = feat.get_hard_neg(i, paper_ids_idx_mapping, all_paper)
        all_paper[i]["hard_neg"] = hard_neg

        return all_paper[i]

    with ProcessPoolExecutor() as executor:
        all_paper = list(
            tqdm(executor.map(get_pos, all_paper), total=len(all_paper))
        )

    logger.info("Extracting Hard Neg")
    with ProcessPoolExecutor() as executor:
        all_paper = list(
            tqdm(
                executor.map(
                    get_hard_neg,
                    range(len(all_paper)),
                ),
                total=len(all_paper),
            )
        )

    # Train, val, test split
    train_paper = list(
        filter(lambda x: train_range[0] <= x["year"] <= train_range[1], all_paper)
    )
    val_paper = list(
        filter(lambda x: val_range[0] <= x["year"] <= val_range[1], all_paper)
    )
    test_paper = list(
        filter(lambda x: test_range[0] <= x["year"] <= test_range[1], all_paper)
    )

    logger.info(f"Num of training paper: {len(train_paper)}")
    logger.info(f"Num of validation paper: {len(val_paper)}")
    logger.info(f"Num of testing paper: {len(test_paper)}")

    with open(f"{args.save_dir}/{dataset_name}_train_test_dataset.json", "w") as f:
        json.dump(
            {   
                "name": dataset_name,
                "train": {
                    data["ids"]: data
                    for data in train_paper
                },
                "valid": {
                    data["ids"]: data
                    for data in val_paper
                },
                "test": {
                    data["ids"]: data
                    for data in test_paper
                }
            },
            f,
            indent=2,
        )
