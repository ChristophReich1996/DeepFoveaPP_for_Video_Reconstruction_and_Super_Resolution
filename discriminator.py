from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    """
    This class implements the discriminator network
    """

    def __init__(self, channels: Tuple[int] = (32, 64, 64, 64, 1), in_channels: int = 3) -> None:
        """
        Constructor method
        :param channels: (Tuple[int]) Number of output channels to be utilized in each separate block
        :param in_channels: (int) Number of input channels (rgb=3)
        """
        # Call super constructor
        super(Discriminator, self).__init__()
        # Init blocks
        self.blocks = nn.ModuleList()
        for index, channel in enumerate(channels):
            if index == 0:
                self.blocks.append(DiscriminatorBlock(in_channels=in_channels, out_channels=channel))
            else:
                self.blocks.append(DiscriminatorBlock(in_channels=channels[index - 1], out_channels=channel))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the module
        :param input: (torch.Tensor) Input sequence of images with shape (batch size, in channels, time, height, width)
        :return: (torch.Tensor) True or false patch prediction (batch size, out channels, 1, height, width)
        """
        # Perform forward pass of main blocks
        for block in self.blocks:
            input = block(input)
        # Perform final average pooling to reduce time dimension
        output = F.adaptive_avg_pool3d(input=input, output_size=(1, input.shape[3], input.shape[4]))
        return output


class FFTDiscriminator(nn.Module):
    """
    This class implements the fft discriminator network
    """

    def __init__(self, channels: Tuple[int] = (32, 64, 64, 64, 1), in_channels: int = 3) -> None:
        """
        Constructor method
        :param channels: (Tuple[int]) Number of output channels to be utilized in each separate block
        :param in_channels: (int) Number of input channels (rgb=3)
        """
        # Call super constructor
        super(FFTDiscriminator, self).__init__()
        # Init blocks
        self.blocks = nn.ModuleList()
        for index, channel in enumerate(channels):
            if index == 0:
                self.blocks.append(DiscriminatorBlock(in_channels=in_channels * 2, out_channels=channel))
            else:
                self.blocks.append(DiscriminatorBlock(in_channels=channels[index - 1], out_channels=channel))
        # Init linear output layer
        self.final_linear = nn.Linear(in_features=256, out_features=1, bias=True)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the module
        :param input: (torch.Tensor) Input sequence of images
        :return: (torch.Tensor) True or false scalar prediction
        """
        assert input.shape[1] == 3, 'Only RGB images are supported!'
        # Perform fft for each feature
        red_fft_features = torch.rfft(input[:, 0].unsqueeze(dim=-1),
                                      signal_ndim=3)[:, :, :, :, 0].permute(0, 4, 1, 2, 3)
        green_fft_features = torch.rfft(input[:, 1].unsqueeze(dim=-1),
                                        signal_ndim=3)[:, :, :, :, 0].permute(0, 4, 1, 2, 3)
        blue_fft_features = torch.rfft(input[:, 2].unsqueeze(dim=-1),
                                       signal_ndim=3)[:, :, :, :, 0].permute(0, 4, 1, 2, 3)
        # Concatenate fft features
        output = torch.cat((red_fft_features, green_fft_features, blue_fft_features), dim=1)
        # Perform forward pass of main blocks
        for block in self.blocks:
            output = block(output)
        # Apply adaptive average pooling to match required shape of linear layer
        output = F.adaptive_avg_pool3d(input=output, output_size=(1, 16, 16))
        # Perform final linear layer
        output = self.final_linear(output.flatten(start_dim=1))
        return output


class DiscriminatorBlock(nn.Module):
    """
    This class implements a residual basic discriminator block, including two 3d convolutions (+ residual mapping) each
    followed by a ELU activation and an 3d average pooling layer at the end. Spectral normalization is utilized in each
    convolution layer.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """
        Constructor method
        :param in_channels: (int) Number of input channels
        :param out_channels: (int) Number of output channels
        """
        # Call super constructor
        super(DiscriminatorBlock, self).__init__()
        # Init main layers
        self.layers = nn.Sequential(
            nn.utils.spectral_norm(
                nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1),
                          stride=(1, 1, 1), bias=True)),
            nn.ELU(),
            nn.utils.spectral_norm(
                nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1),
                          stride=(1, 1, 1), bias=True)),
            nn.ELU(),
        )
        # Init residual mapping
        self.residual_mapping = nn.utils.spectral_norm(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1, 1),
                      padding=(0, 0, 0), stride=(1, 1, 1), bias=True))
        # Init pooling layer
        self.pooling = nn.AvgPool3d(kernel_size=(1, 2, 2))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the block
        :param input: (torch.Tensor) Input tensor of shape (batch size, in channels, time, height, width)
        :return: (torch.Tensor) Output tensor of shape (batch size, out channels, time, height / 2, width / 2)
        """
        # Perform main layers
        output = self.layers(input)
        # Residual mapping
        output = output + self.residual_mapping(input)
        # Perform pooling
        output = self.pooling(output)
        return output


if __name__ == '__main__':
    dis = FFTDiscriminator()
    dis(torch.rand(2, 3, 4, 512, 512))
    exit(11)
