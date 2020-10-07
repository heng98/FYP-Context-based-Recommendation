from paper_data_model import PaperDataModel

import re
import pandas as pd
import os
from collections import defaultdict
from argparse import ArgumentParser

# ABSTRACT_PAT = r'\s?'.join('Abstract') + r'\s+([\s\S]*?)'+ r'\d?\s?' + r'\s?'.join('Introduction')
# O_ABSTRACT_PAT = r'\s?'.join('Abstract') + r'\s+([\s\S]*?)'+ r'\d?\s?' + r'\s?'.join('Overview')
ABSTRACT_PAT = r'\s?'.join('Abstract') + r'\s+([\s\S]*?)'+ r'\d?\s?' + r'(?:' + r'\s?'.join('Introduction') + r'\s?'.join('|Overview)')

ABSTRACT_PAT = re.compile(ABSTRACT_PAT)

def create_csv_file(df_path):
    """Create a csv file containing paper_ids and title,
    sorted lexicographically on paper_ids 
    """
    ids = []
    title = []
    year = []

    with open('../Dataset/aan/release/2014/paper_ids.txt', 'r') as f:
        for line in f.readlines():
            data = line.split('\t')
            ids.append(data[0])
            title.append(data[1])
            year.append(int(data[2]))

    df = pd.DataFrame({'ids': ids, 'title': title, 'year': year})
    df = df[df.ids.str.len().isin([8, 9])]
    df = df.sort_values(by=['ids'])
    df = df.reset_index(drop=True)
    # df.columns = ['paper_ids', 'title']

    df.to_csv(df_path)

def create_paper_to_idx(df_path, dict_path):
    df = pd.read_csv(df_path)
    index = df.index.tolist()
    paper_ids = df['paper_ids'].tolist()

    with open(dict_path, 'a') as f:
        for i, p in zip(index, paper_ids):
            f.write(f'{i}\t{p}\n')

def extract_abstract(paper_path):
    try:
        with open(paper_path, 'r') as f:
            text = f.read()
        
        abstract = ABSTRACT_PAT.findall(text)

        if abstract:
            abstract_text = abstract[0].replace('- \n', '').replace('\n', '')

            return abstract_text
        
        else:
 
            return False


    except FileNotFoundError:
        # print('File not exist')
        return False

def create_dict(dict_path):
    idx_to_paper_ids = dict()
    paper_ids_to_idx = dict()
    with open(dict_path, 'r') as f:
        for line in f.readlines():
            idx, paper_ids = line.split('\t')

            idx = int(idx)

            idx_to_paper_ids[idx] = paper_ids
            paper_ids_to_idx[paper_ids] = idx

    return idx_to_paper_ids, paper_ids_to_idx

def create_citation_network(dict_path):
    _, paper_ids_to_idx = create_dict(dict_path)
    citation_network = defaultdict(list)

    with open('Dataset/aan/release/2014/networks/paper-citation-network-nonself.txt', 'r') as f:
        for line in f.readlines():
            p1, p2 = line.split(' ==> ')
            citation_network[p1].append(paper_ids_to_idx[p2])

    return citation_network

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--path', type=str)
    parser.add_argument('--papers_path', type=str)
    
    arg = parser.parse_args()

    df_path = os.path.join(arg.path, 'paper_title.csv')
    dict_path = os.path.join(arg.path, 'index_paper_ids.txt')

    # if not os.path.exists(df_path):
    create_csv_file(df_path)
    
    create_paper_to_idx(df_path, dict_path)

    df = pd.read_csv(df_path)
    paper_ids = df['paper_ids'].tolist()
    paper_title = df['title'].tolist()
    
    citation_network = create_citation_network(dict_path)

    for p, t in zip(paper_ids, paper_title):
        abstract = extract_abstract(f'{arg.papers_path}/{p}.txt')
        if abstract:
            PaperDataModel(
                title=t, 
                abstract=abstract, 
                year=).to_json(f'{arg.path}/{p}.json')
            


