import torch
from torch import Tensor

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from typing import List, Tuple
from pathlib import Path


class CalibrationError:
    def __init__(
        self,
        predictions: Tensor,
        confidences: Tensor,
        accuracies: Tensor,
        targets: Tensor,
        save_path: str,
        save_png: bool = True,
        num_bins: int = 10,
    ) -> None:
        self.predictions = predictions
        self.confidences = confidences
        self.accuracies = accuracies
        self.targets = targets
        self.save_path = save_path
        self.save_png = save_png
        self.num_bins = num_bins
        self.acc_bin=None
        self.conf_bin=None
        self.prop_bin=None
        self.ce_bin = None
        self.ece = None
        self.mce = None

        self.bin_boundaries = torch.linspace(0, 1, self.num_bins + 1, dtype=torch.float)

    def _sum_by_bin(
        self,
        metric: Tensor,
        index: Tensor,
        count_bin: Tensor,
    ) -> Tensor:
        empty_bin = torch.zeros(self.num_bins)
        met_bin = empty_bin.scatter_add_(dim=0, index=index, src=metric.float())
        met_bin = torch.nan_to_num(met_bin / count_bin)

        return met_bin

    def _complete_binning(
        self
    ) -> Tuple[Tensor, Tensor, Tensor]:
        indices = torch.bucketize(self.confidences, self.bin_boundaries) - 1
        count_bin = torch.bincount(indices, minlength=self.num_bins).float()


        # E_i
        self.conf_bin = self._sum_by_bin(
            metric=self.confidences, index=indices, count_bin=count_bin
        )

        # O_i
        self.acc_bin = self._sum_by_bin(metric=self.accuracies, index=indices, count_bin=count_bin)

        # P(i)
        self.prop_bin = torch.nan_to_num(count_bin / count_bin.sum())

    def _calibration_error_compute(self, norm: str = "l1") -> Tuple[Tensor, Tensor]:
        """
        """
        
        if norm == "l1":
            ce = torch.sum(self.ce_bin * self.prop_bin)
        elif norm == "max":
            ce = torch.max(self.ce_bin)

        return ce.tolist()



