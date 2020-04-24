from typing import List

import torch
import torch.nn as nn
import robust_loss_pytorch


class AdaptiveRobustLoss(nn.Module):
    """
    This class implements the adaptive robust loss function proposed by Jon Barron for image tensors
    """

    def __init__(self, device: str = 'cuda:0', num_of_dimension: int = 1920 * 1080 * 3) -> None:
        """
        Constructor method
        """
        super(AdaptiveRobustLoss, self).__init__()
        # Save parameter
        self.num_of_dimension = num_of_dimension
        # Init adaptive loss module
        self.loss_function = robust_loss_pytorch.AdaptiveLossFunction(num_dims=num_of_dimension, device=device,
                                                                      float_dtype=torch.float)

    def forward(self, prediction: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the loss module
        :param prediction: (torch.Tensor) Prediction
        :param label: (torch.Tensor) Corresponding label
        :return: (torch.Tensor) Scalar loss value
        """
        # Calc difference of the prediction an the label
        loss = prediction - label
        # Reshape loss to use adaptive loss module
        loss = loss.view(-1, self.num_of_dimension)
        # Perform adaptive loss
        loss = self.loss_function.lossfun(loss)
        # Perform mean reduction
        loss = loss.mean()
        return loss


class WassersteinDiscriminatorLoss(nn.Module):
    """
    This class implements the wasserstein loss for a discriminator network
    """

    def __init__(self) -> None:
        """
        Constructor method
        """
        # Call super constructor
        super(WassersteinDiscriminatorLoss, self).__init__()

    def forward(self, prediction_real: torch.Tensor, prediction_fake: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the loss module
        :param prediction_real: (torch.Tensor) Prediction for real samples
        :param prediction_fake: (torch.Tensor) Prediction for fake smaples
        :return: (torch.Tensor) Scalar loss value
        """
        # Compute loss
        loss = torch.mean(prediction_fake) - torch.mean(prediction_real)
        return loss


class WassersteinGeneratorLoss(nn.Module):
    """
    This class implements the wasserstein loss for a generator network
    """

    def __init__(self) -> None:
        """
        Constructor method
        """
        # Call super constructor
        super(WassersteinGeneratorLoss, self).__init__()

    def forward(self, prediction_fake: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the loss module
        :param prediction_fake: (torch.Tensor) Prediction for fake smaples
        :return: (torch.Tensor) Scalar loss value
        """
        # Compute loss
        loss = - torch.mean(prediction_fake)
        return loss


class PerceptualLoss(nn.Module):
    """
    This class implements perceptual loss
    """

    def __init__(self, loss_function: nn.Module = nn.L1Loss(reduction='mean')) -> None:
        """
        Constructor method
        :param loss_function: (nn.Module) Loss function to be utilized to construct the perceptual loss
        """
        # Call super constructor
        super(PerceptualLoss, self).__init__()
        # Save loss function
        self.loss_function = loss_function

    def forward(self, features_prediction: List[torch.Tensor], features_label: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass of the loss module
        :param features_prediction: (List[torch.Tensor]) List of VGG-19 features of the prediction
        :param features_label: (List[torch.Tensor]) List of VGG-19 features of the label
        :return: (torch.Tensor) Scalar loss value
        """
        # Init loss value
        loss = torch.tensor(0.0, dtype=torch.float, device=features_prediction[0].device)
        # Loop over all features
        for feature_prediction, feature_label in zip(features_prediction, features_label):
            # Calc loss and sum up
            loss = loss + self.loss_function(feature_prediction, feature_label)
        # Average loss with number of features
        loss = loss / len(features_prediction)
        return loss
