from typing import Tuple, Union

import os
import json
import torch
import numpy as np


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


def normalize_0_1_batch(input: torch.tensor) -> torch.tensor:
    '''
    Normalize a given tensor to a range of [0, 1]
    Source: https://github.com/ChristophReich1996/Semantic_Pyramid_for_Image_Generation/blob/master/misc.py

    :param input: (Torch tensor) Input tensor
    :return: (Torch tensor) Normalized output tensor
    '''
    input_flatten = input.view(input.shape[0], -1)
    return ((input - torch.min(input_flatten, dim=1)[0][:, None, None, None]) / (
            torch.max(input_flatten, dim=1)[0][:, None, None, None] -
            torch.min(input_flatten, dim=1)[0][:, None, None, None]))


def get_fovea_mask(shape: Tuple[int, int], p_mask: torch.Tensor = None, return_p_mask=True) -> Union[
    torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Function generators a fovea mask for a given probability mask. If no p. mask is given the p. mask is also produced.
    :param shape: (Tuple[int, int]) Shape of the final mask
    :param p_mask: (torch.Tensor) Probability mask
    :param return_p_mask: (bool) If true the probability mask will also be returned
    :return: (Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) Fovea mask and optional p. mask additionally
    """
    if p_mask is None:
        # Get all indexes of image
        indexes = np.stack(np.meshgrid(np.arange(0, shape[1]), np.arange(0, shape[0])), axis=0).reshape((2, -1))
        # Make center point
        center = np.array(
            [np.random.uniform(50, shape[1] - 50), np.random.uniform(50, shape[0] - 50)])
        # Calc euclidean distances
        distances = np.linalg.norm(indexes - center.reshape((2, 1)), ord=2, axis=0)
        # Calc probability mask
        m, b = np.linalg.pinv(np.array([[20, 1], [45, 1]])) @ np.array([[0.98], [0.15]])
        p_mask = np.where(distances < 20, 0.98, 0.0) + np.where(distances > 40, 0.15, 0.0) \
                 + np.where(np.logical_and(distances >= 20, distances <= 40), m * distances + b, 0.0)
        # Probability mask to torch tesnor
        p_mask = torch.from_numpy(p_mask)
    # Make randomized fovea mask
    mask = torch.from_numpy(p_mask.numpy() >= np.random.uniform(low=0, high=1, size=shape[0] * shape[1])).reshape(
        (shape[0], shape[1]))
    if return_p_mask:
        return mask.float(), p_mask.float()
    return mask.float()
