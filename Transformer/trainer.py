import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from transformers import AdamW, get_linear_schedule_with_warmup

from utils import distributed
from model.triplet_loss import TripletLoss

from tqdm import tqdm


class Trainer:
    def __init__(self, model, train_dataset, eval_dataset, args, data_collater=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model.to(self.device)
        if args.distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[args.local_rank])

        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.args = args

        self.data_collater = data_collater

        self.criterion = TripletLoss("l2_norm")

        self.global_step = 0

        if distributed.is_main_process():
            self.tb_writer = SummaryWriter(f"runs/{args.experiment_name}")

    def training_step(self, inputs):
        q = self._prepare_inputs(inputs[0])
        p = self._prepare_inputs(inputs[1])
        n = self._prepare_inputs(inputs[2])

        loss = self.compute_loss(q, p, n)

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        loss.backward()

        return loss.detach()

    def eval_step(self, inputs):
        q = self._prepare_inputs(inputs[0])
        p = self._prepare_inputs(inputs[1])
        n = self._prepare_inputs(inputs[2])

        loss = self.compute_loss(q, p, n)

        loss = loss.mean().detach()

        return loss

    def train(self):
        train_dataloader = self.get_train_dataloader()
        max_steps = (
            len(train_dataloader) // self.args.gradient_accumulation_steps
        ) * self.args.num_epoch

        self._setup_optimizer(max_steps)

        tr_loss = torch.tensor(0.0).to(self.device)
        for epoch in range(self.args.num_epoch):
            self.model.train()
            
            if distributed.is_main_process():
                train_dataloader = tqdm(train_dataloader)

            for step, inputs in enumerate(train_dataloader):
                if ((step + 1) != 0) and self.args.local_rank != -1:
                    with self.model.no_sync():
                        tr_loss += self.training_step(inputs)
                else:
                    tr_loss += self.training_step(inputs)

                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    self.global_step += 1
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                if (self.global_step + 1) % self.args.logging_steps == 0:
                    tr_loss_scalar = tr_loss.item() / self.args.logging_steps
                    tr_loss -= tr_loss
                    self.log({"embedding/train_loss": tr_loss_scalar})

            eval_loss = self.evaluate()
            self.log({"embedding/eval_loss": eval_loss.item()})
            self.save_model(f"weights/{self.args.experiment_name}/weights_{epoch}")

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        eval_dataloader = self.get_eval_dataloader()
        eval_loss = torch.tensor(0.0).to(self.device)
        for inputs in eval_dataloader:
            eval_loss += self.eval_step(inputs)

        if self.args.local_rank != -1:
            eval_loss = distributed.reduce_mean(eval_loss)

        eval_loss = eval_loss / len(eval_dataloader)

        return eval_loss

    def get_train_dataloader(self):
        dataloader = DataLoader(
            self.train_dataset,
            self.args.batch_size,
            collate_fn=self.data_collater,
            pin_memory=True,
            drop_last=True,
            num_workers=1
        )

        return dataloader

    def get_eval_dataloader(self):
        dataloader = DataLoader(
            self.eval_dataset,
            self.args.batch_size,
            collate_fn=self.data_collater,
            pin_memory=True,
            drop_last=True,
            num_workers=1
        )

        if distributed.is_main_process():
            dataloader = tqdm(dataloader)

        return dataloader

    def _prepare_inputs(self, inputs):
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.device)

        return inputs

    def log(self, metrics):
        if distributed.is_main_process():
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self.tb_writer.add_scalar(k, v, self.global_step)

    def save_model(self, path):
        if distributed.is_main_process():
            if hasattr(self.model, "module"):
                model_to_save = self.model.module
            else:
                model_to_save = self.model

            model_to_save.save_pretrained(path)

    def compute_loss(self, q, p, n):
        query_embedding = self.get_embedding(q)
        positive_embedding = self.get_embedding(p)
        negative_embedding = self.get_embedding(n)

        loss = self.criterion(query_embedding, positive_embedding, negative_embedding)

        return loss

    def get_embedding(self, input):
        return self.model(
            input_ids=input["input_ids"],
            attention_mask=input["attention_mask"],
            token_type_ids=input["token_type_ids"]
        )["pooler_output"]

    def _setup_optimizer(self, max_steps):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate)

        warmup_steps = int(max_steps * self.args.warmup_ratio)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, warmup_steps, max_steps
        )
