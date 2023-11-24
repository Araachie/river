import os
from argparse import Namespace as ArgsNamespace
from typing import Any, Dict

import numpy as np
import torch

from dataset import VideoDataset
from evaluation.evaluator import Evaluator
from lutils.configuration import Configuration
from lutils.logger import Logger
from model import Model
from training.trainer import Trainer


def setup_training_arguments(args: ArgsNamespace) -> Dict[str, Any]:
    training_args = dict()

    # Load config file
    training_args["config"] = Configuration(args.config)

    # Other args
    training_args["run_name"] = args.run_name
    training_args["random_seed"] = args.random_seed
    training_args["num_gpus"] = args.num_gpus
    training_args["resume_step"] = args.resume_step
    training_args["use_wandb"] = args.wandb

    return training_args


def training_loop(
        rank: int,
        config: Configuration,
        run_name: str,
        resume_step: int = None,
        cudnn_benchmark: bool = True,
        random_seed: int = None,
        num_gpus: int = 1,
        use_wandb: bool = False):
    # Initialize some stuff
    np.random.seed(random_seed * num_gpus + rank)
    torch.manual_seed(random_seed * num_gpus + rank)
    torch.backends.cudnn.benchmark = cudnn_benchmark
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    # Initialize logger
    extended_run_name = "{}_run-{}".format(config["name"], run_name)
    logger = Logger(
        project="river",
        run_name=extended_run_name,
        use_wandb=use_wandb,
        config=config,
        rank=rank)

    # Load dataset
    logger.info("Loading data")
    datasets = {}

    for key in ["train", "val"]:
        path = os.path.join(config["data"]["data_root"], key)
        datasets[key] = VideoDataset(
            data_path=path,
            input_size=config["data"]["input_size"],
            crop_size=config["data"]["crop_size"],
            frames_per_sample=config["data"]["frames_per_sample"],
            skip_frames=config["data"]["skip_frames"],
            random_horizontal_flip=config["data"]["random_horizontal_flip"],
            aug=config["data"]["aug"],
            albumentations=config["data"]["albumentations"])

    # Setup trainer
    logger.info("Instantiating trainer object")
    num_samples = config["training"]["batching"]["batch_size"] * \
                  config["training"]["optimizer"]["num_training_steps"]
    sampler = torch.utils.data.RandomSampler(
        datasets["train"],
        replacement=True,
        num_samples=num_samples,
        generator=torch.Generator().manual_seed(random_seed * num_gpus + rank))
    trainer = Trainer(
        rank=rank,
        run_name=extended_run_name,
        config=config["training"],
        dataset=datasets["train"],
        sampler=sampler,
        num_gpus=num_gpus,
        device=device)

    # Setup evaluator
    logger.info("Instantiating evaluator object")
    evaluator = Evaluator(
        rank=rank,
        config=config["evaluation"],
        dataset=datasets["val"],
        device=device)

    # Setup model and distribute across gpus
    logger.info("Building the model and distributing it across gpus")
    model = Model(config=config.model)
    model.to(device)
    if (num_gpus > 1) and len(list(model.parameters())) != 0:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[rank],
            broadcast_buffers=False,
            find_unused_parameters=True)

    # Resume training if needed
    if resume_step == -1:
        logger.info("Loading the latest checkpoint")
        trainer.load_checkpoint(model)
    elif resume_step is not None:
        logger.info(f"Loading the checkpoint at step {resume_step}")
        trainer.load_checkpoint(model, f"step_{resume_step}.pth")

    # Launch the training loop
    logger.info("Launching training loop")
    trainer.train(model=model, logger=logger, evaluator=evaluator)
