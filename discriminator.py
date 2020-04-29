from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    """
    This class implements the discriminator network
    """

    def __init__(self, channels: Tuple[int] = (64, 128, 128, 64, 1), in_channels: int = 3, frames: int = 6) -> None:
        """
        Constructor method
        :param channels: (Tuple[int]) Number of output channels to be utilized in each separate block
        :param in_channels: (int) Number of input channels (rgb=3)
        """
        # Call super constructor
        super(Discriminator, self).__init__()
        # Save arguments
        self.in_channels = in_channels
        self.frames = frames
        # Init input block
        self.input_block = InputBlock(channels=frames * in_channels, downscale_factor=4)
        # Init main blocks
        self.blocks = nn.ModuleList()
        for index, channel in enumerate(channels):
            if index == 0:
                self.blocks.append(DiscriminatorBlock(in_channels=in_channels * 2, out_channels=channel))
            else:
                self.blocks.append(DiscriminatorBlock(in_channels=channels[index - 1], out_channels=channel))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the module
        :param input: (torch.Tensor) Input sequence of images with shape (batch size, in channels, time, height, width)
        :return: (torch.Tensor) True or false patch prediction (batch size, out channels, 1, height, width)
        """
        # Perform forward pass of input block
        output = self.input_block(input)
        # Reshape input to match required size for 3d convolutions
        output = output.view(1, self.in_channels * 2, self.frames, output.shape[2], output.shape[3])
        # Perform forward pass of main blocks
        for block in self.blocks:
            output = block(output)
        # Perform final average pooling to reduce time dimension
        output = F.adaptive_avg_pool3d(input=output, output_size=(1, output.shape[3], output.shape[4]))
        return output


class FFTDiscriminator(nn.Module):
    """
    This class implements the fft discriminator network
    """

    def __init__(self, channels: Tuple[int] = (64, 128, 128, 64, 1), in_channels: int = 3, frames: int = 6) -> None:
        """
        Constructor method
        :param channels: (Tuple[int]) Number of output channels to be utilized in each separate block
        :param in_channels: (int) Number of input channels (rgb=3)
        """
        # Call super constructor
        super(FFTDiscriminator, self).__init__()
        # Save arguments
        self.in_channels = in_channels
        self.frames = frames
        # Init input block
        self.input_block = InputBlock(channels=2 * in_channels * frames, downscale_factor=4)
        # Init blocks
        self.blocks = nn.ModuleList()
        for index, channel in enumerate(channels):
            if index == 0:
                self.blocks.append(DiscriminatorBlock(in_channels=in_channels * 2 * 2, out_channels=channel))
            else:
                self.blocks.append(DiscriminatorBlock(in_channels=channels[index - 1], out_channels=channel))
        # Init linear output layer
        self.final_linear = nn.utils.spectral_norm(nn.Linear(in_features=256, out_features=1, bias=True))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the module
        :param input: (torch.Tensor) Input sequence of images
        :return: (torch.Tensor) True or false scalar prediction
        """
        input = input.view(self.frames, self.in_channels, input.shape[2], input.shape[3])
        # Perform fft for each feature. Output shape of (frames, in channels, height, width, real + imag part)
        red_fft_features = torch.rfft(input[:, 0, None].permute([0, 2, 3, 1]), signal_ndim=1).permute([0, 3, 1, 2, 4])
        green_fft_features = torch.rfft(input[:, 1, None].permute([0, 2, 3, 1]), signal_ndim=1).permute([0, 3, 1, 2, 4])
        blue_fft_features = torch.rfft(input[:, 2, None].permute([0, 2, 3, 1]), signal_ndim=1).permute([0, 3, 1, 2, 4])
        # Concatenate fft features
        output = torch.cat((red_fft_features, green_fft_features, blue_fft_features), dim=1).permute(2, 3, 0, 1, 4)
        # Reshape output tensor of fft to match required size for input block
        output = output.contiguous().view(output.shape[0], output.shape[1], -1).permute(2, 0, 1).unsqueeze(dim=0)
        # Perform forward pass of input block
        output = self.input_block(output)
        # Reshape input to match required size for 3d convolutions
        output = output.view(1, self.in_channels * 2 * 2, self.frames, output.shape[2], output.shape[3])
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


class InputBlock(nn.Module):
    """
    This module downscales a given image to a 3d convolution friendly size. This is done by a learnable downsampling
    (2d convolution) and as simple bilinear downsampling. The two outputs gets of the different downsampling operations
    is concatenated in the end.
    """

    def __init__(self, channels: int, downscale_factor: int = 4) -> None:
        """
        Constructor method
        :param downscale_factor: (int) Downsampling factor
        """
        # Call constructor method
        super(InputBlock, self).__init__()
        # Init learnable downsampling operation
        self.learnable_downsampling = nn.Conv2d(in_channels=channels, out_channels=channels,
                                                kernel_size=downscale_factor, stride=downscale_factor, padding=(0, 0),
                                                bias=True)
        # Init bilinear downsampling operation
        self.bilinear_downsampling = nn.Upsample(scale_factor=1 / downscale_factor, mode='bilinear',
                                                 align_corners=False)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        :param input: (torch.Tensor) Input tensor of shape (batch size, channels, height width)
        :return: (torch.Tensor) Output tensor of shape (batch size, 2 * channels, height / down. f., width / down. f.)
        """
        # Perform learnable downsampling
        output_learnable = self.learnable_downsampling(input)
        # Perform normal downsampling
        output_non_learnable = self.bilinear_downsampling(input)
        # Concat outputs
        output = torch.cat((output_learnable, output_non_learnable), dim=1)
        return output


if __name__ == '__main__':
    import time

    # Init discriminator and input
    dis = FFTDiscriminator().cuda()
    input = torch.randn(1, 3 * 6, 1024, 768, device='cuda', dtype=torch.float)
    torch.cuda.synchronize()
    # Test time of forward + backward pass
    start = time.time()
    output = dis(input)
    output = output.sum()
    output.backward()
    torch.cuda.synchronize()
    end = time.time()
    print(end - start)

    # Init discriminator and input
    dis = Discriminator().cuda()
    input = torch.randn(1, 3 * 6, 1024, 768, device='cuda', dtype=torch.float)
    torch.cuda.synchronize()
    # Test time of forward + backward pass
    start = time.time()
    output = dis(input)
    output = output.sum()
    output.backward()
    torch.cuda.synchronize()
    end = time.time()
    print(end - start)
