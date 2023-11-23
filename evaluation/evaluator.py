from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from lutils.configuration import Configuration
from lutils.constants import MAIN_PROCESS
from lutils.dict_wrapper import DictWrapper
from lutils.logger import Logger
from lutils.logging import to_video, make_observations_grid
from lutils.running_average import RunningMean


class Evaluator:
    """
    Class that handles the evaluation
    """

    def __init__(
            self,
            rank: int,
            config: Configuration,
            dataset: Dataset,
            device: torch.device):
        """
        Initializes the Trainer

        :param rank: rank of the current process
        :param config: training configuration
        :param dataset: dataset to train on
        :param sampler: sampler to create the dataloader with
        :param device: device to use for training
        """
        super(Evaluator, self).__init__()

        self.config = config
        self.rank = rank
        self.is_main_process = self.rank == MAIN_PROCESS
        self.device = device

        # Setup dataloader
        self.dataset = dataset
        self.dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.config["batching"]["batch_size"],
            shuffle=True,
            num_workers=self.config["batching"]["num_workers"],
            pin_memory=True)

        # Setup losses
        self.flow_matching_loss = nn.MSELoss()

        self.running_means = RunningMean()

    @torch.no_grad()
    def evaluate(
            self,
            model: nn.Module,
            logger: Logger,
            global_step: int,
            max_num_batches: int = 100):
        """
        Evaluates the model

        """

        if not self.is_main_process:
            return

        if global_step == 0:
            max_num_batches = 10

        model.eval()

        # Setup loading bar
        eval_gen = tqdm(
            self.dataloader,
            total=min(max_num_batches, len(self.dataloader)),
            desc="Evaluation: Batches",
            disable=not self.is_main_process,
            leave=False)
        for i, batch in enumerate(eval_gen):
            if i >= max_num_batches:
                break

            # Fetch data
            observations = batch.cuda()
            num_observations = self.config["num_observations"]
            observations = observations[:, :num_observations]

            # Forward the model
            model_outputs = model(
                observations)

            # Compute the loss
            loss_output = self.calculate_loss(model_outputs)

            # Accumulate scalars
            self.running_means.update(loss_output)

            # Log data only for the 1st batch
            if i != 0:
                continue

            # Log media
            dmodel = model if not isinstance(model, nn.parallel.DistributedDataParallel) else model.module
            model_outputs["generated_observations"] = dmodel.generate_frames(
                observations=observations[:min(4, observations.size(0)), :self.config["condition_frames"]],
                num_frames=self.config["frames_to_generate"],
                steps=100,
                verbose=self.is_main_process)
            self.log_media(model_outputs, logger)

        # Log scalars
        for k, v in self.running_means.get_values().items():
            logger.log(f"Validation/Loss/{k}", v)

        # Finalize logs
        logger.finalize_logs(step=global_step)

        # Close loading bar
        eval_gen.close()

        # Reset the model to train
        model.train()

    @torch.no_grad()
    def calculate_loss(
            self,
            results: DictWrapper[str, Any]) -> DictWrapper[str, Any]:
        """
        Calculates the loss

        :param results: Dict with the model outputs
        :return: [1,] The loss value
        """

        # Flow matching loss
        flow_matching_loss = self.flow_matching_loss(
            results.reconstructed_vectors,
            results.target_vectors)

        # Create auxiliary output
        output = DictWrapper(
            # Loss terms
            flow_matching_loss=flow_matching_loss
        )

        return output

    @staticmethod
    def log_media(results: DictWrapper[str, Any], logger: Logger):
        num_sequences = min(4, results.observations.size(0))

        # Log images grid
        grid = make_observations_grid(
            [
                results.observations,
                results.generated_observations,
            ],
            num_sequences=num_sequences)
        logger.log(f"Validation/Media/reconstructed_observations", logger.wandb().Image(grid))

        # Log real videos
        real_videos = to_video(results.observations[:num_sequences])
        logger.log("Validation/Media/real_videos", logger.wandb().Video(real_videos, fps=7))

        # Log generated videos
        generated_videos = to_video(results.generated_observations)
        logger.log("Validation/Media/generated_videos", logger.wandb().Video(generated_videos, fps=7))
