from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.modulated_deform_conv import ModulatedDeformConvPack


class RecurrentUNet(nn.Module):
    """
    This class implements a recurrent U-Net to perform super resolution base on the DeepFovea architecture
    """

    def __init__(self,
                 channels_encoding: Tuple[Tuple[int, int]] = (
                         (3 * 6, 32), (32, 64), (64, 128), (128, 128), (128, 128)),
                 channels_decoding: Tuple[Tuple[int, int]] = ((384, 128), (384, 128), (256, 64), (112, 16)),
                 channels_super_resolution_blocks: Tuple[Tuple[int, int]] = ((48, 8), (40, 3 * 6))) -> None:
        """
        Constructor method
        :param channels_encoding: (Tuple[Tuple[int, int]]) In and out channels in each encoding path
        :param channels_decoding: (Tuple[Tuple[int, int]]) In and out channels in each decoding path
        :param channels_super_resolution_blocks: (Tuple[Tuple[int, int]]) In and out channels in each s.r. block
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
        # Init final low res output convolution
        self.final_low_res_convolution = nn.Conv2d(in_channels=channels_decoding[-1][1],
                                                   out_channels=channels_super_resolution_blocks[-1][1])
        # Init super-resolution blocks
        self.super_resolution_blocks = nn.ModuleList()
        for index, channel in enumerate(channels_super_resolution_blocks):
            if index == len(channels_super_resolution_blocks) - 1:
                self.super_resolution_blocks.append(
                    SuperResolutionBlock(in_channels=channel[0], out_channels=channel[1], final_output_channels=True))
            else:
                self.super_resolution_blocks.append(
                    SuperResolutionBlock(in_channels=channel[0], out_channels=channel[1]))

    def reset_recurrent_tensor(self) -> None:
        """
        Method resets the recurrent tensor which gets set by calling forward again
        """
        for block in self.decoder_blocks:
            block.reset_recurrent_tensor()

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
                        (F.interpolate(input, scale_factor=2, mode='bilinear', align_corners=False),
                         encoder_activations[-(index + 1)]), dim=1))
            # Normal case
            else:
                output = decoder_block(torch.cat((output, encoder_activations[-(index + 1)]), dim=1))
        # Make low res output
        low_res_output = self.final_low_res_convolution(output)
        # Forward pass of the super resolution blocks
        for index, super_resolution_block in enumerate(self.super_resolution_blocks):
            output = super_resolution_block(
                torch.cat((output, F.interpolate(encoder_activations[0], size=output.shape[2:], mode='bilinear',
                                                 align_corners=False)), dim=1))
        return output + F.interpolate(low_res_output, scale_factor=2 ** len(super_resolution_block),
                                      mode='bilinear'), low_res_output


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
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(3, 3), padding=(1, 1),
                      stride=(1, 1), bias=True),
            nn.ELU(),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), padding=(1, 1),
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
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        # Init previous activation
        self.previous_activation = None

    def reset_recurrent_tensor(self) -> None:
        """
        Method resets the recurrent tensor which gets set by calling forward again
        """
        self.previous_activation = None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        :param input: (torch.Tensor) Input tensor
        :return: (torch.Tensor) Output tensor
        """
        # Init recurrent activation if needed with a random tensor from N(0, 0.02)
        if self.previous_activation is None:
            self.previous_activation = torch.randn((input.shape[0], self.out_channels, input.shape[2], input.shape[3]),
                                                   dtype=torch.float, device=input.device) * 0.02
        # Concatenate previous activation with input
        input = torch.cat((input, self.previous_activation), dim=1)
        # Perform operations
        output = self.convolution_1(input)
        # Init layer norm with shape of input if needed
        if self.layer_norm is None:
            self.layer_norm = nn.LayerNorm(output.shape[1:], elementwise_affine=True)
            # Layer to device
            self.layer_norm.to(self.convolution_1.weight.device)
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


class SuperResolutionBlock(nn.Module):
    """
    This class implements a super resolution block which is used after the original recurrent U-Net
    """

    def __init__(self, in_channels: int, out_channels: int, final_output_channels: int = 3 * 12,
                 final_block: bool = False) -> None:
        """
        Constructor method
        :param in_channels: (int) Number of input channels
        :param out_channels: (int) Number of output channels
        :param final_output_channels: (int) Number of output channels for the mapping to image space
        """
        # Call super constructor
        super(SuperResolutionBlock, self).__init__()
        # Init layers
        self.layers = nn.Sequential(
            ModulatedDeformConvPack(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3),
                                    padding=(1, 1), stride=(1, 1), bias=True),
            nn.ELU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ModulatedDeformConvPack(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3),
                                    padding=(1, 1), stride=(1, 1), bias=True),
            nn.ELU(),
        )
        # Init residual mapping
        self.residual_mapping = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1),
                      padding=(0, 0), stride=(1, 1), bias=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )

        # Init output layer
        self.output_layer = ModulatedDeformConvPack(in_channels=out_channels, out_channels=final_output_channels,
                                                    kernel_size=(1, 1), padding=(0, 0), stride=(1, 1),
                                                    bias=True) if final_block else nn.Identity()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        :param input: (torch.Tensor)
        :return: (Tuple[torch.Tensor, torch.Tensor]) First, output tensor of main convolution. Second, image output
        """
        # Perform main layers
        output = self.layers(input)
        # Perform residual mapping
        output = output + self.residual_mapping(input)
        # Make image output
        output = self.output_layer(output)
        return output
