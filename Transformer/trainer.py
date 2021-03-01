import torch

from transformers import Trainer
from model.triplet_loss import TripletLoss


class TripletTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.criterion = TripletLoss("l2_norm")

    def training_step(self, model, inputs):
        q = self._prepare_inputs(inputs[0])
        p = self._prepare_inputs(inputs[1])
        n = self._prepare_inputs(inputs[2])

        loss = self.compute_loss(model, q, p, n)

        if self.args.n_gpu > 1:
            loss = loss.mean()

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        loss.backward()

        return loss.detach()

    @torch.no_grad()
    def prediction_step(self, model, inputs):
        q = self._prepare_inputs(inputs[0])
        p = self._prepare_inputs(inputs[1])
        n = self._prepare_inputs(inputs[2])

        loss = self.compute_loss(model, q, p, n)

        loss = loss.mean().detach()

        return (loss, None, None)

    def compute_loss(self, model, q, p, n):
        query_embedding = model(q)
        positive_embedding = model(p)
        negative_embedding = model(n)

        loss = self.criterion(query_embedding, positive_embedding, negative_embedding)

        return loss