import torch
from torch import Tensor

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from typing import List, Tuple

import os
from pathlib import Path

import numpy.typing as npt


class PatternDetection:
    """
    A class for performing pattern detection using PCA, t-SNE and K-means clustering algorithms on input data.

    Args:
        fp_data (Tensor): The input data tensor.
        fp_target (Tensor): The target tensor.
        class_labels (List[str]): A list of class labels.
        save_path (str): The path to save the output files.
        save_prefix (str): The prefix to use for the output file names.
        save_png (bool, optional): Whether to save the output plots in PNG format. Defaults to True.

    Attributes:
        fp_data (Tensor): The input data tensor.
        fp_target (Tensor): The target tensor.
        class_labels (List[str]): A list of class labels.
        save_path (str): The path to save the output files.
        save_prefix (str): The prefix to use for the output file names.
        save_png (bool): Whether to save the output plots in PNG format.
        pca_values (Tensor): The PCA reduced values.
        tsne_values (Tensor): The t-SNE reduced values.
        k_means_predictions (Tensor): The K-means predictions.

    Methods:
        _pca(self, n_components: int = 2) -> None:
            Applies PCA on the input data and stores the reduced values.
        _tsne(self, n_components: int = 2, **kwargs) -> None:
            Applies t-SNE on the input data and stores the reduced values.
        _k_means(self, n_clusters: int = 3, **kwargs) -> Tensor:
            Applies K-means clustering on the PCA reduced values and returns the predictions.
        _plot_cluster(self, dat: npt.NDArray, classes: str, algorithm: str) -> None:
            Plots the reduced data with target classes and saves the plot in SVG format.
        cluster(self) -> None:
            Performs PCA, t-SNE and K-means clustering, plots and saves the output files.
    """
    def __init__(
        self,
        fp_data: Tensor,
        fp_target: Tensor,
        class_labels: List[str],
        save_path: str,
        save_prefix: str,
        save_png: bool = True,
    ) -> None:
        """
        Initializes the PatternDetection class.

        Args:
            fp_data (Tensor): The input data tensor.
            fp_target (Tensor): The target tensor.
            class_labels (List[str]): A list of class labels.
            save_path (str): The path to save the output files.
            save_prefix (str): The prefix to use for the output file names.
            save_png (bool, optional): Whether to save the output plots in PNG format. Defaults to True.
        """
        self.fp_data = fp_data
        self.fp_target = fp_target
        self.class_labels = class_labels
        self.save_path = save_path
        self.save_prefix = save_prefix
        self.save_png = save_png

    def _pca(self, n_components: int = 2) -> None:
        """
        Applies PCA on the input data and stores the reduced values.

        Args:
            n_components (int, optional): The number of principal components to keep. Defaults to 2.
        """
        self.pca_values = PCA(n_components=n_components).fit_transform(
            self.fp_data.numpy()
        )

    def _tsne(self, n_components: int = 2, **kwargs) -> None:
        """
        Applies t-SNE (t-distributed stochastic neighbor embedding) dimensionality reduction technique 
        to the fingerprint data and saves the result in self.tsne_values.

        Parameters:
            n_components (int): Number of components in the reduced space. Default is 2.
            **kwargs: Additional keyword arguments to be passed to the TSNE constructor.

        Returns:
            None
        """
        x_data = self.fp_data.numpy()
        self.tsne_values = TSNE(n_components=n_components, **kwargs).fit_transform(
            x_data
        )

    def _k_means(self, n_clusters: int = 3, **kwargs) -> Tensor:
        """
        Performs K-Means clustering on the fingerprint data after PCA dimensionality reduction
        and saves the result in self.k_means_predictions.

        Parameters:
            n_clusters (int): Number of clusters to form. Default is 3.
            **kwargs: Additional keyword arguments to be passed to the KMeans constructor.

        Returns:
            Tensor: The K-Means clustering result.
        """
        self.k_means_predictions = Tensor(
            KMeans(n_clusters=n_clusters, **kwargs).fit_transform(self.pca_values)
        )

        _, self.k_means_predictions = self.k_means_predictions.max(dim=1)

    def _plot_cluster(self, dat: npt.NDArray, classes: str, algorithm: str) -> None:
        """
        Plots the clusters in a scatter plot using Seaborn library and saves the plot in a file.

        Parameters:
            dat (npt.NDArray): The data to be plotted.
            classes (str): A list of target classes.
            algorithm (str): The name of the dimensionality reduction algorithm used for clustering.

        Returns:
            None
        """
        df = pd.DataFrame(
            {
                "x": dat[:, 0],
                "y": dat[:, 1],
                "target": classes,
            }
        )

        plt.clf()
        sns.set(rc={"figure.figsize": (15, 10)})

        pca_fig = sns.scatterplot(data=df, x="x", y="y", hue="target")
        sns.move_legend(pca_fig, "upper left", bbox_to_anchor=(1, 1))
        pca_fig.set_title(algorithm + " Dimension Reduction")
        fig = pca_fig.get_figure()
        plt.tight_layout()
        if self.save_png:
            fig.savefig(
                os.path.join(
                    self.save_path, self.save_prefix + "_" + algorithm + ".svg"
                ),
                dpi=400,
                format="svg",
            )
        else:
            fig.show()

    def cluster(self) -> None:
        """
        Performs PCA and t-SNE dimensionality reduction on the fingerprint data, and plots the clusters
        using the _plot_cluster method.

        Parameters:
            None

        Returns:
            None
        """
        self._pca(n_components=2)
        self._tsne(n_components=2)
        # self._k_means(n_clusters=4)

        class_key = dict((v, k) for k, v in self.class_labels.items())
        classes = [class_key.get(item, item) for item in self.fp_target.tolist()]

        pca_x = self.pca_values
        tnse_x = self.tsne_values

        Path(self.save_path).mkdir(parents=True, exist_ok=True)

        self._plot_cluster(dat=pca_x, classes=classes, algorithm="PCA")
        self._plot_cluster(dat=tnse_x, classes=classes, algorithm="TSNE")
