import torch
from torch import Tensor

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from typing import List, Tuple
from pathlib import Path


class CalibrationError:
    """
    Class to compute calibration error of a model's confidence predictions.
    Args:
        predictions (Tensor): Predicted labels by the model
        confidences (Tensor): Confidence scores for the predictions
        accuracies (Tensor): Binary tensor of whether the prediction was correct
        targets (Tensor): Ground truth labels
        save_path (str): Path to save the calibration graph
        save_png (bool): Whether to save the calibration graph as png
        num_bins (int): Number of bins to divide the confidence scores into

    Attributes:
        predictions (Tensor): Predicted labels by the model
        confidences (Tensor): Confidence scores for the predictions
        accuracies (Tensor): Binary tensor of whether the prediction was correct
        targets (Tensor): Ground truth labels
        save_path (str): Path to save the calibration graph
        save_png (bool): Whether to save the calibration graph as png
        num_bins (int): Number of bins to divide the confidence scores into
        bin_boundaries (Tensor): Boundaries of confidence score bins
        conf_bin (Tensor): Mean confidence score for each bin
        acc_bin (Tensor): Mean accuracy score for each bin
        prop_bin (Tensor): Proportion of samples in each bin
        ce_bin (Tensor): Calibration error for each bin
        ece (float): Expected Calibration Error
        mce (float): Max Calibration Error
    """
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
        """
        Initialize the CalibrationError class.

        Args:
            predictions (Tensor): Predicted labels by the model
            confidences (Tensor): Confidence scores for the predictions
            accuracies (Tensor): Binary tensor of whether the prediction was correct
            targets (Tensor): Ground truth labels
            save_path (str): Path to save the calibration graph
            save_png (bool): Whether to save the calibration graph as png
            num_bins (int): Number of bins to divide the confidence scores into

        Returns:
            None
        """
        self.predictions = predictions
        self.confidences = confidences
        self.accuracies = accuracies
        self.targets = targets
        self.save_path = save_path
        self.save_png = save_png
        self.num_bins = num_bins


        self.bin_boundaries = torch.linspace(0, 1, self.num_bins + 1, dtype=torch.float)

    def _sum_by_bin(
        self,
        metric: Tensor,
        index: Tensor,
        count_bin: Tensor,
    ) -> Tensor:
        """
        Compute the sum of a given metric for each bin.

        Args:
            metric (Tensor): Metric to compute sum for (confidence or accuracy)
            index (Tensor): Bin index for each sample
            count_bin (Tensor): Number of samples in each bin

        Returns:
            Tensor: Sum of metric for each bin
        """
        empty_bin = torch.zeros(self.num_bins)
        met_bin = empty_bin.scatter_add_(dim=0, index=index, src=metric.float())
        met_bin = torch.nan_to_num(met_bin / count_bin)

        return met_bin

    def _complete_binning(self) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Completes the binning process by creating the bins for the confidences, 
        accuracies, and proportions. It uses the bucketize function to group the 
        confidences into bins and calculates the number of instances in each bin, 
        E_i, O_i, and P(i) respectively.

        Returns:
            Tuple of Tensors: Tuple containing confidences, accuracies, and 
                            proportions for each bin.
        """
        indices = torch.bucketize(self.confidences, self.bin_boundaries) - 1
        count_bin = torch.bincount(indices, minlength=self.num_bins).float()

        # E_i
        self.conf_bin = self._sum_by_bin(
            metric=self.confidences, index=indices, count_bin=count_bin
        )

        # O_i
        self.acc_bin = self._sum_by_bin(
            metric=self.accuracies, index=indices, count_bin=count_bin
        )

        # P(i)
        self.prop_bin = torch.nan_to_num(count_bin / count_bin.sum())

    def _calibration_error_compute(self, norm: str = "l1") -> Tuple[Tensor, Tensor]:
        """
        Calculates calibration error using L1 or maximum norm.
        
        Args:
            norm (str): The norm to be used in calculation. Default is "l1".

        Returns:
            Tuple of Tensors: Tuple containing the calibration error using the 
                            specified norm and the tensor converted to a list.
        """

        if norm == "l1":
            ce = torch.sum(self.ce_bin * self.prop_bin)
        elif norm == "max":
            ce = torch.max(self.ce_bin)

        return ce.tolist()

    def produce_results(self) -> None:
        """
        Produces the results of the calibration model by calling the _complete_binning
        and _calibration_error_compute methods, and creating a calibration graph using
        seaborn. Saves the graph to disk if save_png is True, otherwise shows the graph.

        Returns:
            None
        """
        with torch.no_grad():
            self._complete_binning()

        self.ce_bin = torch.abs(self.acc_bin - self.conf_bin)

        self.ece = self._calibration_error_compute(norm="l1")
        self.mce = self._calibration_error_compute(norm="max")

        print(f"Expected Calibration Error: {str(round(self.ece,4))}")
        print(f"Max Calibration Error: {str(round(self.mce,4))}")

        x_axis = ["%.2f" % elem for elem in self.bin_boundaries.tolist()]
        x_axis = [f"{x_axis[i]}-{x_axis[i+1]}" for i in range(len(x_axis) - 1)]

        df = pd.DataFrame({"Confidence Bins": x_axis, "ECE": self.ce_bin * self.prop_bin})

        

        Path(self.save_path).mkdir(parents=True, exist_ok=True)

        plt.clf()
        # sns.set(rc={"figure.figsize": (15, 10)})

        ce_fig = sns.barplot(data=df, x="Confidence Bins", y="ECE")
        ce_fig.set_title("Calibration Error")

        fig = ce_fig.get_figure()
        plt.xticks(rotation=90)
        plt.tight_layout()
        if self.save_png:
            fig.savefig(self.save_path + "calibration_graph.svg", dpi=400, format="svg")
        else:
            fig.show()

        plt.clf()

