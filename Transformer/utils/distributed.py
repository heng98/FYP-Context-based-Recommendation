import torch
import os
import torch.distributed as dist

def init_distributed_mode(config):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        config.rank = int(os.environ["RANK"])
        config.world_size = int(os.environ['WORLD_SIZE'])
        config.gpu = int(os.environ['LOCAL_RANK'])

        config.distributed = True
    else:
        config.distributed = False
        return

    torch.cuda.set_device(config.gpu)

    dist.init_process_group(backend='nccl')

def reduce_mean(tensor):
    rt = tensor.detach().clone()
    dist.reduce(rt, 0)
    rt /= dist.get_world_size()

    return rt

def is_main_process():
    if not dist.is_initialized():
        return True
    else:
        return dist.get_rank() == 0
