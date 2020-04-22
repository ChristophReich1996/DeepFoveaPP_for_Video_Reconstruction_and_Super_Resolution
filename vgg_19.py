from typing import List, Tuple

import torch
import torch.nn as nn
import torchvision


class VGG19(nn.Module):
    """
    This class implements a pre-trained vgg-19 classification network which returns the activations of
    the first five conv2 layers
    """

    def __init__(self, indexes_of_layers_to_return_features: Tuple[int] = (2, 7, 12, 21, 30)) -> None:
        """
        Constructor method
        :param indexes_of_layers_to_return_features: (Tuple[int]) Layers to return feature output
        """
        # Call super constructor
        super(VGG19, self).__init__()
        # Load pre-trained feature part of vgg-19 from torchvision
        self.vgg_19_features = torchvision.models.vgg19(pretrained=True).features
        # Convert feature module into model list of layers
        self.vgg_19_features = nn.ModuleList(list(self.vgg_19_features))
        # Save parameter
        self.indexes_of_layers_to_return_features = indexes_of_layers_to_return_features

    def forward(self, input: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass
        :param input: (torch.Tensor) Input image
        :return: (List[torch.Tenor]) List of features
        """
        # Init list to store features. Could also be implemented with hooks but sometimes problematic with data parallel
        features = []
        # Perform forward pass of each layer in vgg-19 feature path
        for index, layer in enumerate(self.vgg_19_features):
            # Perform forward pass of layer
            input = layer(input)
            # Save output of layer is chosen
            if index in self.indexes_of_layers_to_return_features:
                # Feature saved to list
                features.append(input)
        return features
