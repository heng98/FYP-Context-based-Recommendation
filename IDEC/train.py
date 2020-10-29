from model import IDEC
from trainer import Trainer
from argparse import ArgumentParser
from dataset import NewsGroupDataset

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--n_cluster', type=int, default=20)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--tol', type=float, default=0.1)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--update_interval', type=int, default=5)
    parser.add_argument('--channel', type=str, default='2000,4000,2000,500,50')
    
    config = parser.parse_args()

    model = IDEC(config)
    model.from_pretrained('./weights/pretrain_weight.pth')

    data = NewsGroupDataset()

    trainer = Trainer(model, data, config)
    trainer.train()