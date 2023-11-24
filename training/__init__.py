from argparse import Namespace as ArgsNamespace

from lutils.distributed import setup_torch_distributed
from .training_loop import setup_training_arguments, training_loop


def train(rank: int, args: ArgsNamespace, temp_dir: str):
    setup_torch_distributed(rank, args, temp_dir)

    # Execute training loop
    training_args = setup_training_arguments(args=args)
    training_loop(rank=rank, **training_args)
