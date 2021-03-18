import torch
import numpy as np
from tqdm import tqdm

def to_device_dict(d, device):
    return {k: v.to(device) for k, v in d.items()}


@torch.no_grad()
def embed_documents(model, dataset, tokenizer, device, save_path=None, embedding_size=768):
    # print(dataset.values())
    doc_embedding_vectors = torch.empty(len(dataset), embedding_size)
    for i, data in enumerate(tqdm(dataset.values())):
        encoded = tokenizer(
            data["title"],
            data["abstract"],
            padding="max_length",
            max_length=256,
            truncation=True,
            return_tensors="pt",
        )

        encoded = to_device_dict(encoded, device)

        query_embedding = model(encoded)
        doc_embedding_vectors[i] = query_embedding

    if save_path:
        torch.save(doc_embedding_vectors, save_path)

    return doc_embedding_vectors.cpu().numpy()