import torch
from torch import Tensor

from torchvision import models
from torchvision import datasets

from tqdm import tqdm
from typing import Tuple, List

import glob
import pickle

import os
import shutil
from pathlib import Path
from itertools import compress


class TorchModel:
    """
    A class that represents a TorchModel object, which consists of a pre-trained ConvNext model from torchvision, a dataset
    of images to use for training or inference, and a set of data transformations to be applied to the images before
    feeding them into the model.

    Args:
        data_path (str): A string containing the path to the dataset.
        model_size (str): A string containing the name of the model size.

    Attributes:
        data_path (str): The path to the directory containing the dataset.
        model_size (str): The size of the ConvNext model to use (e.g., 'small', 'medium', or 'large').
        model (torchvision.models.ConvNet): The pre-trained ConvNext model.
        dataset (torchvision.datasets.ImageFolder): The dataset of images to use for training or inference.
        data_transform (torchvision.transforms.Compose): The set of data transformations to be applied to the images.
    """

    def __init__(self, data_path: str, model_path:str,  model_size: str) -> None:
        """
        Initializes a TorchModel instance.

        Args:
            data_path: A string containing the path to the dataset.
            model_size: A string containing the name of the model size.

        Returns:
            None
        """
        # Model Size
        self.model_size = model_size #.capitalize()
        self.data_path = os.path.abspath(data_path)
        self.model_path = os.path.abspath(model_path)

        # Data Transformation
        # Using the bespoke transformation of ConvNext
        transform_path = glob.glob( os.path.join(self.model_path, self.model_size, "transform", "*.pickle"))
        transform_files_num = len(transform_path)
        if transform_files_num == 0:
            raise ValueError("No transformation function pickle found.")
        elif transform_files_num > 1:
            raise ValueError("Multiple transformation function pickles found.")
        
        with open(transform_path[0], 'rb') as handle:
            self.data_transform = pickle.load(handle)

        # self.data_transform = models.get_weight(
        #     f"ConvNeXt_{self.model_size}_Weights.DEFAULT"
        # ).transforms

        # Data Loader
        # Using torchvison.models.ImageFolder
        self.dataset = datasets.ImageFolder(data_path, transform=self.data_transform())

        # Model
        # Accessing ConvNext from torchvision
        model_path = glob.glob( os.path.join(self.model_path, self.model_size, "model", "*.pt"))
        model_files_num = len(model_path)
        if model_files_num == 0:
            raise ValueError("No Model .pt file found.")
        elif model_files_num > 1:
            raise ValueError("Multiple todel .pt files found.")
        self.model =  torch.load(model_path[0])

        # self.model = models.get_model(
        #     f"convnext_{self.model_size}", num_classes=len(self.dataset.classes)
        # )
        # self.model.eval()

        
    @property
    def num_classes(self) -> int:
        """
        Returns the number of classes in the dataset.

        Returns:
            An integer representing the number of classes.
        """
        return len(self.dataset.classes)

    @property
    def class_labels(self) -> List[str]:
        """
        Returns the class labels in the dataset.

        Returns:
            A list of strings containing the class labels.
        """
        return self.dataset.classes

    @property
    def targets(self) -> Tensor:
        """
        Returns the targets in the dataset.

        Returns:
            A tensor containing the targets.
        """
        return Tensor(self.dataset.targets)
    
    @property
    def data(self) -> Tensor:
        """
        Returns the data in the dataset.

        Returns:
            A tensor containing the data.
        """
        dat, _ = zip(*self.dataset)
        dat = torch.cat([x.unsqueeze(0) for x in dat])
        return(dat)

        

class Inference(TorchModel):
    """
    A class that represents an Inference object, which is a subclass of TorchModel. It inherits the dataset, model, and
    data transformations from TorchModel, and adds the ability to run inference on the model and save false positives.

    Attributes:
        results_path (str): The path to the directory where the results will be saved.
        fp_path (str): The path to the directory where the false positives will be saved.
    """
    
    def __init__(
        self, data_path: str, model_path:str, model_size: str, results_path: str, fp_folder: str
    ) -> None:
        """
        Initializes an Inference instance.

        Args:
            data_path: A string containing the path to the dataset.
            model_size: A string containing the name of the model size.
            results_path: A string containing the path to the results directory.
            fp_folder: A string containing the name of the folder for false positives.

        Returns:
            None
        """

        super().__init__(data_path= data_path, model_path = model_path, model_size = model_size)

        # Results path
        self.results_path = results_path
        self.fp_path = os.path.join(self.results_path, fp_folder)


    def infer(self) -> None:
        """
        Runs inference on the model.

        Returns:
            None
        """
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
        """
        Returns the class probabilities for the false positives.

        Returns:
            A tensor containing the class probabilities.
        """
        return self.class_probs[torch.logical_not(self.accuracies)]

    @property
    def fp_targets(self) -> Tensor:
        """
        Returns the targets for the false positives.

        Returns:
            A tensor containing the targets.
        """
        return Tensor(self.dataset.targets)[torch.logical_not(self.accuracies)]
