import torch
import re


class Ranker:
    """Currently for simple reranker"""

    def __init__(self, reranker_model, doc_embedding_vectors, device):
        self.reranker_model = reranker_model
        self.doc_embedding_vectors = doc_embedding_vectors
        self.device = device

        self.pattern = re.compile(r"[\w']+")


    def rank(self, query_embedding, candidates, query_data, candidate_data):
        candidates_idx = [c[0] for c in candidates]
        query_embedding = query_embedding.expand(len(candidates), -1).to(self.device)
        candidates_embedding = torch.from_numpy(
            self.doc_embedding_vectors[candidates_idx]
        ).to(self.device)

        cos_similarity = torch.nn.functional.cosine_similarity(
            query_embedding, candidates_embedding
        ).unsqueeze(0).T

        jaccard = self._jaccard(
            [query_data["abstract"]] * len(candidate_data),
            [c["abstract"] for c in candidate_data]
        )
        jaccard = torch.tensor(jaccard, device=self.device).unsqueeze(0).T
        

        confidence = self.reranker_model(cos_similarity, jaccard)
        confidence = confidence.flatten().tolist()

        reranked_candidates = [(c["ids"], conf) for c, conf in zip(candidate_data, confidence)]
        
        return sorted(reranked_candidates, key=lambda x: x[1], reverse=True)

    def _jaccard(self, text_1, text_2):
        result = []
        tokenized_t1 = [self.pattern.findall(t1) for t1 in text_1]
        tokenized_t2 = [self.pattern.findall(t2) for t2 in text_2]

        for t1, t2 in zip(tokenized_t1, tokenized_t2):
            union = len(set(t1).union(set(t2)))
            if union > 0:
                result.append(
                    len(set(t1).intersection(set(t2))) / union
                )
            else:
                result.append(0)

        return result


class TransformerRanker:
    def __init__(self, reranker_model, device, tokenizer):
        self.reranker_model = reranker_model
        self.device = device
        self.tokenizer = tokenizer

    def rank(self, query, candidates):
        query_text = query["title"] + query["abstract"]
        candidates_text = [c["title"] + c["abstract"] for c in candidates]
        candidates_ids = [c["ids"] for c in candidates]

        query_encoded = self._encode(query_text)
        candidates_encoded = self._encode(candidates_text)
        candidates_encoded["input_ids"][:, 0] = self.tokenizer.sep_token_id

        for k in query_encoded:
            query_encoded[k] = query_encoded[k].expand_as(candidates_encoded[k])

        combined_encoded = {
            k: torch.cat([query_encoded[k], candidates_encoded[k]], axis=1).to(self.device) 
            for k in query_encoded
        }

        similarity = torch.flatten(torch.sigmoid(self.reranker_model(**combined_encoded)["logits"]))
        sorted_sim, indices = torch.sort(similarity, descending=True)

        sorted_sim = sorted_sim.tolist()
        indices = indices.tolist()

        reranked_candidates = [
            (candidates_ids[idx], sim) for idx, sim in zip(indices, sorted_sim)
        ]

        return reranked_candidates

    def _encode(self, text):
        return self.tokenizer(
            text,
            padding="max_length",
            max_length=256,
            truncation=True,
            return_tensors="pt",
        )

        