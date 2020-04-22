from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RecurrentUNet(nn.Module):
    """
    This class implements a recurrent U-Net to perform super resolution base on the DeepFovea architecture
    """

    def __init__(self,
                 channels_encoding: Tuple[Tuple[int, int]] = ((3, 32), (32, 64), (64, 128), (128, 128), (128, 128)),
                 channels_decoding: Tuple[Tuple[int, int]] = ((384, 128), (384, 128), (256, 64), (128, 32)),
                 channels_final_block: Tuple[Tuple[int, int]] = ((32, 32), (32, 3))) -> None:
        """
        Constructor method
        :param channels_encoding: (Tuple[Tuple[int, int]]) In and out channels in each encoding path
        :param channels_decoding: (Tuple[Tuple[int, int]]) In and out channels in each decoding path
        """
        # Call super constructor
        super(RecurrentUNet, self).__init__()
        # Init decoder blocks
        self.encoder_blocks = nn.ModuleList()
        for channel in channels_encoding:
            self.encoder_blocks.append(
                ResidualBlock(in_channels=channel[0], out_channels=channel[1]))
        # Init decoder blocks
        self.decoder_blocks = nn.ModuleList()
        for channel in channels_decoding:
            self.decoder_blocks.append(TemporalBlock(in_channels=channel[0], out_channels=channel[1]))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        :param input: (torch.Tensor) Input frame
        :return: (torch.Tensor) Super resolution output frame
        """
        # Init list to store encoder outputs
        encoder_activations = []
        # Forward pass of encoder blocks
        for index, encoder_block in enumerate(self.encoder_blocks):
            input = encoder_block(input)
            if index != len(self.encoder_blocks) - 1:
                encoder_activations.append(input)
        # Forward pass of decoder blocks
        for index, decoder_block in enumerate(self.decoder_blocks):
            # Bottleneck output case
            if index == 0:
                output = decoder_block(
                    torch.cat(
                        (F.upsample(input, scale_factor=2, mode='bilinear'), encoder_activations[-(index + 1)]), dim=1))
            # Normal case
            else:
                output = decoder_block(torch.cat((output, encoder_activations[-(index + 1)]), dim=1))
        return output


class ResidualBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """
        Constructor method
        :param in_channels: (int) Number of input channels
        :param out_channels: (int) Number of output channels
        """
        # Call super constructor
        super(ResidualBlock, self).__init__()
        # Init main layers
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), padding=(1, 1),
                      stride=(1, 1), bias=True),
            nn.ELU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), padding=(1, 1),
                      stride=(1, 1), bias=True),
            nn.ELU()
        )
        # Init residual mapping
        self.residual_mapping = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1),
                                          padding=(0, 0), stride=(1, 1), bias=True) \
            if in_channels != out_channels else nn.Identity()
        # Init pooling operation
        self.pooling = nn.AvgPool2d(kernel_size=(2, 2))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        :param input: (torch.Tensor) Input tensor of shape (batch size, in channels, height, width)
        :return: (torch.Tensor) Output tensor of shape (batch size, out channels, height / 2, width / 2)
        """
        # Forward pass main layers
        output = self.layer(input)
        # Residual mapping
        output = output + self.residual_mapping(input)
        # Perform pooling
        output = self.pooling(output)
        return output


class TemporalBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """
        Constructor method
        :param in_channels: (int) Number of input channels
        :param out_channels: (int) Number of output channels
        """
        # Call super constructor
        super(TemporalBlock, self).__init__()
        # Save number of output channels for residual activation
        self.out_channels = out_channels
        # Init layer
        self.convolution_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                       kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), bias=True)
        self.layer_norm = None
        self.activation_1 = nn.ELU()
        self.convolution_2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3),
                                       padding=(1, 1), stride=(1, 1), bias=True)
        self.activation_2 = nn.ELU()
        # Init residual mapping
        self.residual_mapping = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                          kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)
        # Init upsampling layer
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        # Init previous activation
        self.previous_activation = None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        :param input: (torch.Tensor) Input tensor
        :return: (torch.Tensor) Output tensor
        """
        # Init recurrent activation if needed
        if self.previous_activation is None:
            self.previous_activation = torch.ones((input.shape[0], self.out_channels, input.shape[2], input.shape[3]),
                                                  dtype=torch.float, device=input.device)
        # Concatenate previous activation with input
        input = torch.cat((input, self.previous_activation), dim=1)
        # Perform operations
        output = self.convolution_1(input)
        # Init layer norm with shape of input if needed
        if self.layer_norm is None:
            self.layer_norm = nn.LayerNorm(output.shape[1:], elementwise_affine=True)
        # Perform layer norm
        output = self.layer_norm(output)
        # Save activation as previous activation
        self.previous_activation = output.detach().clone()
        # Perform rest of operations
        output = self.convolution_2(output)
        output = self.activation_2(output)
        # Perform residual mapping
        output = output + self.residual_mapping(input)
        # Perform upsampling
        output = self.upsample(output)
        return output


if __name__ == '__main__':
    unet = RecurrentUNet()
    print(sum([p.numel() for p in unet.parameters()]))
    output = unet(torch.rand(1, 3, 256, 256))
    print(output.shape)
    exit(22)
