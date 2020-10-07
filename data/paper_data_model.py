from dataclasses import dataclass
from typing import List
import json

@dataclass(frozen=True)
class PaperDataModel:
    title: str
    abstract: str
    citations: List[str]
    year: int

    def _to_dict(self):
        data = dict()
        data['title'] = self.title
        data['abstract'] = self.abstract
        data['citations'] = self.citations
        data['year'] = self.year

        return data

    def to_json(self, path):
        json.dump(self._to_dict(), open(path, 'w'), indent=4)