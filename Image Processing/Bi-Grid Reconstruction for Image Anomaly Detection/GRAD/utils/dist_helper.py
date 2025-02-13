import os
import subprocess

import torch
import torch.distributed as dist

def setup_distributed(port):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(
            backend='nccl',
            init_method=f'tcp://127.0.0.1:{port}',
            rank=rank,
            world_size=world_size
        )
        return rank, world_size
    else:
        return 0, 1
