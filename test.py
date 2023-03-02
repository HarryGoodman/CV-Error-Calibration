import argparse
import os
from pathlib import Path

from typing import List

from torch import classes

from src.Inference import Inference
from src.ConfusionMatrix import ConfusionMatrix
from src.CalibrationError import CalibrationError
from src.pattern_detection import PatternDetection
from src.false_postive import FalsePostiveImage


class ArgumentParser(argparse.ArgumentParser):
    """
    A subclass of the argparse.ArgumentParser used for the Calibration Script.

    Parameters:
    model_size (List[str]): A list of strings containing the available model sizes to choose from.
    default_model_size (str): A string representing the default model size to use if not specified.

    Methods:
    __init__(self, model_size: List[str], default_model_size: str) -> None:
        Initializes an instance of ArgumentParser with the specified model size and default model size.

        Parameters:
        model_size (List[str]): A list of strings containing the available model sizes to choose from.
        default_model_size (str): A string representing the default model size to use if not specified.

        Returns:
        None
    """

    def __init__(self, model_size: List[str,], default_model_size: str) -> None:
        """
        Initializes an instance of ArgumentParser with the specified model size and default model size.

        Parameters:
        model_size (List[str]): A list of strings containing the available model sizes to choose from.
        default_model_size (str): A string representing the default model size to use if not specified.

        Returns:
        None
        """
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
        self.add_argument("data")
        self.add_argument("--model", choices=model_size, default=default_model_size)



model_size = ["tiny", "small", "base", "large"]
default_model_size = "tiny"


def main(args=None):
    """
    The main function of the Calibration Script which runs the inference, confusion matrix, calibration error and false positive
    analysis functions.

    Parameters:
    args (Optional[Namespace]): A Namespace object containing the arguments for the Calibration Script. Defaults to None.

    Returns:
    None
    """
    parser = ArgumentParser(
        model_size=model_size,
        default_model_size=default_model_size,
    )
    args = parser.parse_args(args)
    args.data = os.path.abspath(args.data)

    results_path = os.path.join(os.path.dirname(args.data), "results", "")
    model_path = os.path.join(os.path.dirname(args.data), "models")


    inf = Inference(
        data_path=args.data,
        model_path = model_path,
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
        data_path=os.path.join(
            os.path.dirname(args.data), "results", "false_positives"
        ),
        model_path = model_path,
        model_size=args.model,
        save_path=results_path,
        save_prefix="RGB",
    )
    fpi.analyse()

    cp_cluster = PatternDetection(
        fp_data=inf.fp_class_probs,
        fp_target=inf.fp_targets,
        class_labels=inf.dataset.class_to_idx,
        save_path=results_path,
        save_prefix="ClassProb",
        save_png=True,
    )
    cp_cluster.cluster()


if __name__ == "__main__":
    SystemExit(main())
