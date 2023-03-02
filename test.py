import argparse
import os
from pathlib import Path

from typing import List

from torch import classes

from src.Inference import Inference
from src.ConfusionMatrix import ConfusionMatrix
from src.CalibrationError import CalibrationError
from src.false_postive import FalsePostiveImage


class ArgumentParser(argparse.ArgumentParser):
    """
    Argument parser used for this Calibration Script
    """

    def __init__(self, model_size: List[str,], default_model_size: str) -> None:
        super().__init__(
            description=(
                """
                Compute Calibration Error, Confusion Matrix\
                 and assess patterns in false positives for the specified model\
                 and size (Currently only for handle ConvNext).
                """
            ),
            add_help=True,
            usage=f"python test.py data [--model {{ {', '.join(model_size)} }}]",
        )
        self.add_argument("data", default=test_folder)
        self.add_argument("--model", choices=model_size, default=default_model_size)


test_folder = "../CIFAR-10-images_small/test/"
model_size = ["tiny", "small", "base", "large"]
default_model_size = "tiny"


def main(args=None):
    parser = ArgumentParser(
        model_size=model_size,
        default_model_size=default_model_size,
    )
    args = parser.parse_args(args)
    args.data = os.path.abspath(args.data)

    results_path = os.path.join(os.path.dirname(args.data) , "results", '')

    inf = Inference(
        data_path=args.data,
        model_size=args.model,
        results_path=results_path,
        fp_folder="false_positives",
    )
    inf.infer()

    cm = ConfusionMatrix(
        predictions=inf.predictions,
        targets=inf.targets,
        class_labels=inf.class_labels,
        save_path=results_path,
        save_png=True,
    )
    cm.plot_conf_matrix()


    ce = CalibrationError(
        predictions=inf.predictions,
        confidences=inf.confidences,
        accuracies=inf.accuracies,
        targets=inf.targets,
        save_path=results_path,
        save_png=True,
        num_bins=10,
    )
    ce.produce_results()


    fpi = FalsePostiveImage(
        data_path = os.path.join(os.path.dirname(args.data) , "results", 'false_positives'), 
        model_size= args.model, 
        save_path = results_path,
        save_prefix = "RGB",
    ) 
    fpi.analyse()

if __name__ == "__main__":
    SystemExit(main(args=[test_folder]))
