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


class PatternDetection:
    def __init__(
        self,
        fp_data: Tensor,
        fp_target: Tensor,
        class_labels: List[str],
        save_path: str,
        save_prefix: str,
        save_png: bool = True,
    ) -> None:
        self.fp_data = fp_data
        self.fp_target = fp_target
        self.class_labels = class_labels
        self.save_path = save_path
        self.save_prefix = save_prefix
        self.save_png = save_png

    def _pca(self, n_components: int = 2) -> None:
        self.pca_values = Tensor(
            PCA(n_components=n_components).fit_transform(self.fp_data.numpy())
        )

    def _tsne(self, n_components: int = 2, **kwargs) -> None:
        self.tsne_values = Tensor(
            TSNE(n_components=n_components, **kwargs).fit_transform(
                self.fp_data.numpy()
            )
        )

    def _k_means(self, n_clusters: int = 3, **kwargs) -> Tensor:
        self.k_means_predictions = Tensor(KMeans(
            n_clusters=n_clusters, **kwargs
        ).fit_transform(self.pca_values))

        _, self.k_means_predictions = self.k_means_predictions.max(dim=1)


    def cluster(self) -> None:
        self._pca(n_components= 2)
        # self._tsne(n_components = 2)
        self._k_means(n_clusters = 3)

        pca_x = self.pca_values.numpy()

        df = pd.DataFrame({"x": pca_x[:,0], "y": pca_x[:,1], "cluster": self.k_means_predictions, "target": self.fp_target})

        Path(self.save_path).mkdir(parents=True, exist_ok=True)

        plt.clf()

        sns.set(rc={"figure.figsize": (15, 10)})

        pca_fig = sns.scatterplot(data = df, x = "x", y = "y", hue = "target", style = "cluster")
        pca_fig.set_title("K-means Clustering")
        fig = pca_fig.get_figure()
        if self.save_png:
            fig.savefig(os.path.join(self.save_path, self.save_prefix[0] + "_k_means.svg"), dpi=400, format="svg")
        else:
            fig.show()

        plt.clf()
