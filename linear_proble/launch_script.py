import os
import torch
import torch.distributed as dist


def setup():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)


def main():
    setup()


if __name__ == "__main__":
    main()
