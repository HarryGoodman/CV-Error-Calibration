import argparse

from typing import List

from torch import classes

from src.Inference import Inference
from src.ConfusionMatrix import ConfusionMatrix


class ArgumentParser(argparse.ArgumentParser):
    """
    Argument parser used for this Calibration Script
    """

    def __init__(self, model_size: List[str,], default_model_size: str) -> None:
        super().__init__(
            description=(
                "Compute Calibration Error, Confusion Matrix"
                " and assess patterns in false positives for the specified model"
                " and size (Currently only for handle ConvNext)."
            ),
            add_help=False,
            usage=f"python test.py data [--model {{ {', '.join(model_size)} }}]",
        )
        self.add_argument("data")
        self.add_argument("--model", choices=model_size, default=default_model_size)


model_size = ["tiny", "small", "base", "large"]
default_model_size = "tiny"

if __name__ == "__main__":
    parser = ArgumentParser(
        model_size=model_size,
        default_model_size=default_model_size,
    )
    args = parser.parse_args()

    inf = Inference(data_path=args.data, model_size=args.model)

    inf.infer()

    cm = ConfusionMatrix(
        predictions=inf.get_predictions(),
        targets= inf.get_true_target(),
        classes = inf.get_class_labels(),
        save_path = args.data + "results/"
    )

    cm.plot_conf_matrix()

