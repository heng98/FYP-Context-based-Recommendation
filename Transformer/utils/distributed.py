import torch
import os
import torch.distributed as dist

def init_distributed(config):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        config.rank = int(os.environ["RANK"])
        config.world_size = int(os.environ['WORLD_SIZE'])
        config.gpu = int(os.environ['LOCAL_RANK'])

    torch.cuda.set_device(config.gpu)

    dist.init_process_group(backend='nccl')

def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt)
    rt /= nprocs

    return rt
