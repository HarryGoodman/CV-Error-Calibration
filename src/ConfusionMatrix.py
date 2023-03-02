import torch
from torch import Tensor

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from typing import List
from pathlib import Path

class ConfusionMatrix:
    """
    A class for generating and plotting a confusion matrix based on predicted and actual targets.

    Args:
        predictions (Tensor): A tensor of predicted targets.
        targets (Tensor): A tensor of actual targets.
        class_labels (List[str]): A list of class labels.
        save_path (str): The path to save the confusion matrix plot.
        save_png (bool, optional): Whether to save the confusion matrix plot as a PNG file. Default is True.

    Attributes:
        predictions (Tensor): A tensor of predicted targets.
        targets (Tensor): A tensor of actual targets.
        save_path (str): The path to save the confusion matrix plot.
        class_labels (List[str]): A list of class labels.
        save_png (bool): Whether to save the confusion matrix plot as a PNG file.
        conf_matrix (DataFrame): A pandas dataframe of the confusion matrix.

    Methods:
        generate_conf_matrix() -> Tensor:
            Generates the confusion matrix based on the predicted and actual targets.
            Returns the confusion matrix as a tensor.
        plot_conf_matrix() -> None:
            Plots the confusion matrix as a heatmap and saves the plot to the specified path.
    """
    def __init__(
        self,
        predictions: Tensor,
        targets: Tensor,
        class_labels: List[str],
        save_path: str,
        save_png: bool = True,
    ) -> None:
        """
        Constructor for the ConfusionMatrix class.

        Args:
        - predictions (Tensor): tensor of predicted classes.
        - targets (Tensor): tensor of true classes.
        - class_labels (List[str]): list of class labels.
        - save_path (str): path to save the confusion matrix.
        - save_png (bool): whether to save the plot as a PNG file or not.

        Returns:
        None
        """
        self.predictions = predictions
        self.targets = targets
        self.save_path = save_path
        self.class_labels = class_labels
        self.save_png = save_png

    def generate_conf_matrix(
        self,
    ) -> Tensor:
        """
        Generates the confusion matrix.

        Args:
        None

        Returns:
        Tensor: the confusion matrix.
        """
        classes_n = len(self.class_labels)
        self.conf_matrix = torch.zeros(classes_n, classes_n)

        for pred_class in range(classes_n):
            self.conf_matrix[pred_class] = torch.bincount(
                self.predictions[self.targets == pred_class], minlength=classes_n
            ).float()

        self.conf_matrix = pd.DataFrame(
            self.conf_matrix.numpy(),
            index=[i for i in self.class_labels],
            columns=[i for i in self.class_labels],
        )

    def plot_conf_matrix(self) -> None:
        """
        Plots the confusion matrix.

        Args:
        None

        Returns:
        None
        """
        self.generate_conf_matrix()
        with pd.option_context("display.max_rows", None, "display.max_columns", None):
            print("Confusion Matrix:\n")
            print(self.conf_matrix)

        Path(self.save_path).mkdir(parents=True, exist_ok=True)

        plt.clf()
        sns.set(rc={"figure.figsize": (15, 8)})

        conf_fig = sns.heatmap(
            self.conf_matrix,
            annot=True,
        ).set_title("Confusion Matrix")

        fig = conf_fig.get_figure()
        plt.tight_layout()
        if self.save_png:
            fig.savefig(self.save_path + "confusion_matrix.svg", dpi=400, format="svg")
        else:
            fig.show()

        plt.clf()

        
