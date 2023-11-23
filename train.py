import tempfile
from argparse import ArgumentParser, Namespace as ArgsNamespace

import torch

from training import train


def parse_args() -> ArgsNamespace:
    parser = ArgumentParser()
    parser.add_argument("--run-name", type=str, required=True, help="Name of the current run.")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file.")
    parser.add_argument("--num-gpus", type=int, default=None, help="Number of gpus to use for training."
                                                                   "By default uses all available gpus.")
    parser.add_argument("--resume-step", type=int, default=None, help="Step to resume the training from.")
    parser.add_argument("--random-seed", type=int, default=1543, help="Random seed.")
    parser.add_argument("--wandb", action="store_true", help="If defined, use wandb for logging.")

    return parser.parse_args()


def main():
    args = parse_args()

    # Count gpus
    if args.num_gpus is None:
        args.num_gpus = torch.cuda.device_count()

    # Launch processes.
    print('Launching processes...')
    torch.multiprocessing.set_start_method('spawn')
    with tempfile.TemporaryDirectory() as temp_dir:
        if args.num_gpus == 1:
            train(rank=0, args=args, temp_dir=temp_dir)
        else:
            torch.multiprocessing.spawn(fn=train, args=(args, temp_dir), nprocs=args.num_gpus)


if __name__ == "__main__":
    main()
