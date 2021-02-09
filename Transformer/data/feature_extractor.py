import json
from typing import List, Dict, Any
import json
import logging

from collections import Counter
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
    feat = FeatureExtractor()
    path = "./dblp_dataset.json"
    papers = json.load(open(path, "r"))

    candidate_paper = list(filter(lambda x: (x["year"] > 2014) and (len(x["citations"])>7), papers["papers"]))
    
    paper_ids_idx_mapping = feat.build_paper_ids_idx_mapping(candidate_paper)

    def get_pos(p):
        p["pos"] = list(set(p.pop("citations")) & set(paper_ids_idx_mapping))
        return p

    def get_hard_neg(i):
        hard_neg = feat.get_hard_neg(i, paper_ids_idx_mapping, candidate_paper)
        candidate_paper[i]["hard_neg"] = hard_neg

        return candidate_paper[i]

    with ProcessPoolExecutor() as executor:
        candidate_paper = list(executor.map(get_pos, tqdm(candidate_paper)))
    logger.info("Extracting Hard Neg")
    with ProcessPoolExecutor() as executor:
        candidate_paper = list(executor.map(get_hard_neg, tqdm(range(len(candidate_paper)))))

    train_paper = list(filter(lambda x: (x["year"] <= 2016), candidate_paper))
    test_paper = list(filter(lambda x: (x["year"] > 2016), candidate_paper))

    train_mapping = feat.build_paper_ids_idx_mapping(train_paper)
    test_mapping = feat.build_paper_ids_idx_mapping(test_paper)



    with open("./dblp_train_test_dataset.json", "w") as f:
        json.dump(
            {   
                "mapping": paper_ids_idx_mapping,
                "train": train_paper,
                "test": test_paper
            },  
            f,
            indent=2,
        )
