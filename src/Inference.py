import torch
from torchvision import models
from torchvision import datasets

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

        # Confindence and Predictions
        self.confidences = self.predictions = []
        

    def infer(self) -> None:
        for data, _ in self.dataset:
            data = data.unsqueeze(0)
            with torch.no_grad():
                logits = self.model(data)
                softmax = torch.exp(logits) / torch.sum(torch.exp(logits))
                output = torch.max(softmax, dim=-1)

            confidence = output.values.item()
            self.confidences.append(confidence)

            prediction = output.indices.item()
            self.predictions.append(prediction)

        print(self.confidences)
