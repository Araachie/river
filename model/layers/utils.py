import torch
import torch.nn as nn

from lutils.tensor_folder import TensorFolder


class SequenceConverter(nn.Module):
    def __init__(self, backbone: nn.Module):
        super(SequenceConverter, self).__init__()

        self.backbone = backbone

    @staticmethod
    def convert(x, n):
        if isinstance(x, list):
            return [TensorFolder.fold(e, n) for e in x]
        elif x.dim() <= 1:
            return x
        return TensorFolder.fold(x, n)

    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        assert len(args) > 0

        observations_count = args[0].size(1)
        for sequences in args:
            assert sequences.size(1) == observations_count, "Incompatible observations count"

        xs = [TensorFolder.flatten(sequences) for sequences in args]
        x = self.backbone(*xs)

        if isinstance(x, dict):
            for k, v in x.items():
                x[k] = self.convert(v, observations_count)
        else:
            x = self.convert(x, observations_count)

        return x
