import torch
from transformers import AutoTokenizer

from typing import NoReturn, List, Union, Dict, Any, Set
import json
import logging

logging.basicConfig(level = logging.INFO)
logger = logging.getLogger()
# logger.setLevel(logging.INFO)
# console_handler = logging.StreamHandler()
# console_handler.setLevel(logging.INFO)
# logger.addHandler(console_handler)

class FeatureExtractor():
    def __init__(self, pretrained_path: str) -> NoReturn:
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_path, use_fast=True)    

    def get_input(self, title: Union[str, List[str]], abstract: Union[str, List[str]]) -> Dict[str, torch.Tensor]:
        """Tokenization for all titles and abstracts

        Args:
        """
        data = self.tokenizer(title, abstract, padding='max_length', max_length=512, truncation=True, return_tensors='pt')
        return data

    @staticmethod
    def get_pos(paper: Dict[str, Any], papers_ids_set: Set[str]) -> Dict[str, Any]:
        d = {
            'pos': list(set(paper['citations']).intersection(papers_ids_set))
        }

        return d
    
    @staticmethod
    def get_hard_neg(query_paper: str, papers: Dict[str, Dict[str, Any]]) -> List[str]:
        papers_set = set(papers.keys())
        try:
            # Choose the positive that is in papers
            pos = set(papers[query_paper]['pos']).intersection(papers_set)
            hard_neg = set()
            # Get postive of positive of query paper
            for p in pos:
                hard_neg.update(papers[p]['pos'])
            
            # Remove positive paper inside hard negative
            # Choose hard negative only inside papers
            hard_neg = (hard_neg - pos).intersection(papers_set)

            return list(hard_neg)

        except KeyError:
            logger.error(f'Abstract is not in paper with ids {p}')


if __name__ == '__main__':
    feat = FeatureExtractor('allenai/scibert_scivocab_cased')
    path = './processed_aan_data/dataset.json'
    papers = json.load(open(path, 'r'))


    ##TODO feature extractor for test
    # for mode in ['train', 'test']:
    paper_ids = []
    titles = []
    abstracts = []
    network = dict()

    # sequence of paper is labelled here
    for paper in papers['train']:
        paper_ids.append(paper['ids'])
        titles.append(paper['title'])
        abstracts.append(paper['abstract'])

    # Batch encode all the titles and abstracts
    encoded = feat.get_input(titles, abstracts)
    paper_ids_set = set(paper_ids)

    # Get all the positive from dataset
    for paper in papers['train']:
        network[paper['ids']] = feat.get_pos(paper, paper_ids_set)

    # Get all the hard negative from network
    for p in paper_ids:
        network[p]['hard'] = feat.get_hard_neg(p, network)
    
    torch.save({
        'paper_ids': paper_ids,
        'encoded': encoded.data,
        'network': network
    }, f"train_file.pth")
        
