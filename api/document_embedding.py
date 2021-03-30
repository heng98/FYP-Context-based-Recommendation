from transformers import AutoTokenizer, AutoModel
import torch

class DocumentEmbeddingModel:
    def __init__(self, model_path):
        self.model = AutoModel.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained("allenai/cs_roberta_base")
        
        self.device = torch.device("cuda:0")
        self.model = self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def embed(self, title, abstract):
        encoded_query = self.tokenizer(
            title,
            abstract,
            padding="max_length",
            max_length=256,
            truncation=True,
            return_tensors="pt",
        )
        print(encoded_query["input_ids"])
        encoded_query = {k: v.to(self.device) for k, v in encoded_query.items()}
        query_embedding = self.model(**encoded_query)[
            "last_hidden_state"
        ][:, 0]

        return query_embedding.cpu().numpy()[0]