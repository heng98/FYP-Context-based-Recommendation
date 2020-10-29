from dataclasses import dataclass
from typing import List
import json

@dataclass(frozen=True)
class PaperDataModel:
    ids : str
    title: str
    abstract: str
    citations: List[str]
    year: int

    def to_dict(self):
        data = dict()
        data['ids'] = self.ids
        data['title'] = self.title
        data['abstract'] = self.abstract
        data['citations'] = self.citations
        data['year'] = self.year

        return data

    def to_json(self, path):
        json.dump(self.to_dict(), open(path, 'w'), indent=2)

    @classmethod
    def from_json(cls, json_file):
        data = json.load(open(json_file, 'r'))

        return cls(data['ids'], data['title'], data['abstract'], data['citations'], data['year'])
