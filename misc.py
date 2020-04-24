import os
import json
import torch


class Logger(object):
    """
    Class to log different metrics
    Source: https://github.com/ChristophReich1996/Semantic_Pyramid_for_Image_Generation
    """

    def __init__(self) -> None:
        self.metrics = dict()
        self.hyperparameter = dict()

    def log(self, metric_name: str, value: float) -> None:
        """
        Method writes a given metric value into a dict including list for every metric
        :param metric_name: (str) Name of the metric
        :param value: (float) Value of the metric
        """
        if metric_name in self.metrics:
            self.metrics[metric_name].append(value)
        else:
            self.metrics[metric_name] = [value]

    def save_metrics(self, path: str) -> None:
        """
        Static method to save dict of metrics
        :param metrics: (Dict[str, List[float]]) Dict including metrics
        :param path: (str) Path to save metrics
        :param add_time_to_file_name: (bool) True if time has to be added to filename of every metric
        """
        # Save dict of hyperparameter as json file
        with open(os.path.join(path, 'hyperparameter.txt'), 'w') as json_file:
            json.dump(self.hyperparameter, json_file)
        # Iterate items in metrics dict
        for metric_name, values in self.metrics.items():
            # Convert list of values to torch tensor to use build in save method from torch
            values = torch.tensor(values)
            # Save values
            torch.save(values, os.path.join(path, '{}.pt'.format(metric_name)))

def psnr(prediction: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    """
    Function computes the Peak Signal to Noise Ratio
    PSNR = 10 * log10(max[y]**2 / MSE(y, y'))
    Source: https://github.com/ChristophReich1996/CellFlowNet
    :param prediction: (torch.Tensor) Prediction
    :param label: (torch.Tensor) Label
    :return: (torch.Tensor) PSNR value
    """
    assert prediction.numel() == label.numel(), 'Prediction tensor and label tensor must have the number of elements'
    return 10.0 * torch.log10(prediction.max() ** 2 / (torch.mean((prediction - label) ** 2) + 1e-08))


def ssim(prediction: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    """
    Function computes the structural similarity
    SSMI = 10 * log10(max[y]**2 / MSE(y, y'))
    Source: https://github.com/ChristophReich1996/CellFlowNet
    :param prediction: (torch.Tensor) Prediction
    :param label: (torch.Tensor) Label
    :return: (torch.Tensor) SSMI value
    """
    assert prediction.numel() == label.numel(), 'Prediction tensor and label tensor must have the number of elements'
    # Calc means and vars
    prediction_mean = prediction.mean()
    prediction_var = prediction.var()
    label_mean = label.mean()
    label_var = label.var()
    # Calc correlation coefficient
    correlation_coefficient = (1 / label.numel()) * torch.sum((prediction - prediction_mean) * (label - label_mean))
    return ((2.0 * prediction_mean * label_mean) * (2.0 * correlation_coefficient)) / \
           ((prediction_mean ** 2 + label_mean ** 2) * (prediction_var + label_var))