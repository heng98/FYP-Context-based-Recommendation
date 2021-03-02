import torch
import os
import torch.distributed as dist

def init_distributed_mode(args):
    if args.local_rank != -1:
        args.distributed = True
    else:
        args.distributed = False
        return

    args.train_triplets_per_epoch //= dist.get_world_size()
    args.eval_triplets_per_epoch //= dist.get_world_size()

    torch.cuda.set_device(args.local_rank)

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
