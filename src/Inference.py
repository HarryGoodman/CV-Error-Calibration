import torch
from torch import Tensor

from torchvision import models
from torchvision import datasets

from tqdm import tqdm
from typing import Tuple, List

import os
import shutil
from pathlib import Path
from itertools import compress


class TorchModel:
    def __init__(self, data_path: str, model_size: str) -> None:
        # Model Size
        self.model_size = model_size.capitalize()
        self.data_path = os.path.abspath(data_path)

        # Data Transformation
        # Using the bespoke transformation of ConvNext
        self.data_transform = models.get_weight(
            f"ConvNeXt_{self.model_size}_Weights.DEFAULT"
        ).transforms

        # Data Loader
        # Using torchvison.models.ImageFolder
        self.dataset = datasets.ImageFolder(data_path, transform=self.data_transform())

        # Model
        # Accessing ConvNext from torchvision
        self.model = models.get_model(
            f"convnext_{self.model_size}", num_classes=len(self.dataset.classes)
        )
        self.model.eval()

        
    @property
    def num_classes(self) -> int:
        """
        Get the number of classes (produced by ImageFolder).
        Returns:
            number of class labels
        """
        return len(self.dataset.classes)

    @property
    def class_labels(self) -> List[str]:
        """
        Get the class labels (produced by ImageFolder).
        Returns:
            List of the class labels (preserving order of index)
        """
        return self.dataset.classes

    @property
    def targets(self) -> Tensor:
        return Tensor(self.dataset.targets)
    
    @property
    def data(self) -> Tensor:
        dat, _ = zip(*self.dataset)
        dat = torch.cat([x.unsqueeze(0) for x in dat])
        return(dat)

        

class Inference(TorchModel):
    def __init__(
        self, data_path: str, model_size: str, results_path: str, fp_folder: str
    ) -> None:

        super().__init__(data_path= data_path, model_size = model_size)

        # Results path
        self.results_path = results_path
        self.fp_path = os.path.join(self.results_path, fp_folder)


    def infer(self) -> None:
        dat = self.data

        with torch.no_grad():
            output = self.model(dat)
            self.class_probs = torch.exp(output) / torch.sum(torch.exp(output))

        self.confidences, self.predictions = self.class_probs.max(dim=1)
        self.accuracies = Tensor(self.predictions.eq(self.targets))
        self.confidences = self.confidences.float()

        fp_classes = list(compress(self.dataset.targets, torch.logical_not(self.accuracies).tolist()))
        class_key = dict((v, k) for k, v in self.dataset.class_to_idx.items())
        fp_classes = [class_key.get(item, item) for item in fp_classes]

        file_paths, _ = zip(*self.dataset.imgs)
        fp_file_paths = list(compress(file_paths, torch.logical_not(self.accuracies).tolist()))
        fp_file_name = [os.path.basename(file_x) for file_x in fp_file_paths]

        fp_save_path = [
            os.path.join(self.fp_path, class_x, file_x)
            for class_x, file_x in zip(fp_classes, fp_file_name)
        ]

        for subfolder in self.dataset.classes:
            subfolder_path = os.path.join(self.fp_path, subfolder)
            Path(subfolder_path).mkdir(parents=True, exist_ok=True)

        for file_path, save_path in zip(fp_file_paths, fp_save_path):
            _ = shutil.copy2(file_path, save_path)

    @property
    def fp_class_probs(self) -> Tensor:
        return self.class_probs[torch.logical_not(self.accuracies)]

    @property
    def fp_target(self) -> Tensor:
        return Tensor(self.dataset.targets)[torch.logical_not(self.accuracies)]
