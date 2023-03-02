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

from src.Inference import TorchModel
from src.pattern_detection import PatternDetection

class FalsePostiveImage(TorchModel):
    def __init__(
        self, 
        data_path: str, 
        model_size: str, 
        save_path: str,
        save_prefix: str,
        save_png: bool = True
    ) -> None:

        super().__init__(data_path= data_path, model_size = model_size)

        # Save details
        self.save_path = save_path
        self.save_prefix = save_prefix,
        self.save_png = save_png

    def analyse(self):
        
        
        dat = self.data.flatten(1,3)

        pattern_detection = PatternDetection(
            fp_data = dat,
            fp_target = self.targets,
            class_labels = self.dataset.class_to_idx,
            save_path = self.save_path,
            save_prefix = self.save_prefix[0],
            save_png = self.save_png
        )

        pattern_detection.cluster()



