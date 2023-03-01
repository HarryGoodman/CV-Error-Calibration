import torch
from torch import Tensor

from torchvision import models
from torchvision import datasets

from tqdm import tqdm
from typing import Tuple, List


class Inference:
    def __init__(self, data_path: str, model_size: str) -> None:
        # Model Size
        self.model_size = model_size.capitalize()

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

    def infer(self) -> None:
        data, self.targets = zip(*self.dataset)
        self.targets = Tensor(self.targets)
        data = torch.cat([x.unsqueeze(0) for x in data])

        with torch.no_grad():
            output = self.model(data)
            self.class_probs = torch.exp(output) / torch.sum(torch.exp(output))

        self.confidences, self.predictions = self.class_probs.max(dim=1)
        self.accuracies = self.predictions.eq(self.targets)
        self.confidences = self.confidences.float()


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
    def fp_class_probs(self) -> Tensor:
        return self.class_probs[torch.logical_not(self.accuracies)]

    @property
    def fp_target(self) -> Tensor:
        return Tensor(self.dataset.targets)[torch.logical_not(self.accuracies)]
