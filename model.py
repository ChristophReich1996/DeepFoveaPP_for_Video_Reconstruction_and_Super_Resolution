from typing import Tuple, Union, Type

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
                        (F.interpolate(input, scale_factor=2, mode='bilinear', align_corners=False),
                         encoder_activations[-(index + 1)]), dim=1))
            # Normal case
            else:
                output = decoder_block(torch.cat((output, encoder_activations[-(index + 1)]), dim=1))
        # Forward pass of the super resolution blocks
        for index, super_resolution_block in enumerate(self.super_resolution_blocks):
            output = super_resolution_block(
                torch.cat((output, F.interpolate(encoder_activations[0], size=output.shape[2:], mode='bilinear',
                                                 align_corners=False)), dim=1))
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


class AxialAttention3d(nn.Module):
    """
    This class implements the axial attention operation for 3d volumes.
    """

    def __init__(self, in_channels: int, out_channels: int, dim: int, span: int, groups: int = 8) -> None:
        """
        Constructor method
        :param in_channels: (int) Input channels to be employed
        :param out_channels: (int) Output channels to be utilized
        :param dim: (int) Dimension attention is applied to (0 = height, 1 = width, 2 = depth)
        :param span: (int) Span of attention to be used
        :param groups: (int) Multi head attention groups to be used
        """
        # Call super constructor
        super(AxialAttention3d, self).__init__()
        # Check parameters
        assert (in_channels % groups == 0) and (out_channels % groups == 0), \
            "In and output channels must be a factor of the utilized groups."
        assert dim in [0, 1, 2], "Illegal argument for dimension"
        # Save parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dim = dim
        self.span = span
        self.groups = groups
        self.group_channels = out_channels // groups
        # Init initial query, key and value mapping
        self.query_key_value_mapping = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=2 * out_channels, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm1d(num_features=2 * out_channels, track_running_stats=True, affine=True)
        )
        # Init output normalization
        self.output_normalization = nn.BatchNorm1d(num_features=2 * out_channels, track_running_stats=True, affine=True)
        # Init similarity normalization
        self.similarity_normalization = nn.BatchNorm2d(num_features=3 * self.groups, track_running_stats=True,
                                                       affine=True)
        # Init embeddings
        self.relative_embeddings = nn.Parameter(torch.randn(2 * self.group_channels, 2 * self.span - 1),
                                                requires_grad=True)
        relative_indexes = torch.arange(self.span, dtype=torch.long).unsqueeze(dim=1) \
                           - torch.arange(self.span, dtype=torch.long).unsqueeze(dim=0) \
                           + self.span - 1
        self.register_buffer("relative_indexes", relative_indexes.view(-1))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        :param input: (torch.Tensor) Input tensor of the shape [batch size, in channels, h, w, d]
        :return: (torch.Tensor) Output tensor of the shape [batch size, out channels, h, w, d]
        """
        # Reshape input dependent on the dimension to be utilized
        if self.dim == 0:  # Attention over volume height
            input = input.permute(0, 3, 4, 1, 2)  # [batch size, width, depth, in channels, height]
        elif self.dim == 1:  # Attention over volume width
            input = input.permute(0, 2, 4, 1, 3)  # [batch size, height, depth, in channels, width]
        else:  # Attention over volume depth
            input = input.permute(0, 2, 3, 1, 4)  # [batch size, height, width, in channels, depth]
        # Save shapes
        batch_size, dim_1, dim_2, channels, dim_attention = input.shape
        # Reshape tensor to the shape [batch size * dim 1 * dim 2, channels, dim attention]
        input = input.reshape(batch_size * dim_1 * dim_2, channels, dim_attention).contiguous()
        # Perform query, key and value mapping
        query_key_value = self.query_key_value_mapping(input)
        # Split tensor to get the query, key and value tensors
        query, key, value = query_key_value \
            .reshape(batch_size * dim_1 * dim_2, self.groups, self.group_channels * 2, dim_attention) \
            .split([self.group_channels // 2, self.group_channels // 2, self.group_channels], dim=2)
        # Get all embeddings
        embeddings = self.relative_embeddings.index_select(dim=1, index=self.relative_indexes) \
            .view(2 * self.group_channels, self.span, self.span)
        # Split embeddings
        query_embedding, key_embedding, value_embedding = \
            embeddings.split([self.group_channels // 2, self.group_channels // 2, self.group_channels], dim=0)
        # Apply embeddings to query, key and value
        query_embedded = torch.einsum("bgci, cij -> bgij", query, query_embedding)
        key_embedded = torch.einsum("bgci, cij -> bgij", key, key_embedding)
        # Matmul between query and key
        query_key = torch.einsum("bgci, bgcj -> bgij", query_embedded, key_embedded)
        # Construct similarity map
        similarity = torch.cat([query_key, query_embedded, key_embedded], dim=1)
        # Perform normalization
        similarity = self.similarity_normalization(similarity) \
            .view(batch_size * dim_1 * dim_2, 3, self.groups, dim_attention, dim_attention).sum(dim=1)
        # Apply softmax
        similarity = F.softmax(similarity, dim=3)
        # Calc attention map
        attention_map = torch.einsum("bgij, bgcj->bgci", similarity, value)
        # Calc attention embedded
        attention_map_embedded = torch.einsum("bgij, cij->bgci", similarity, value_embedding)
        # Construct output
        output = torch.cat([attention_map, attention_map_embedded], dim=-1) \
            .view(batch_size * dim_1 * dim_2, 2 * self.out_channels, dim_attention)
        # Final output batch normalization
        output = self.output_normalization(output).view(batch_size, dim_1, dim_2, self.out_channels, 2,
                                                        dim_attention).sum(dim=-2)
        # Reshape output back to original shape
        if self.dim == 0:  # [batch size, width, depth, in channels, height]
            output = output.permute(0, 3, 4, 1, 2)
        elif self.dim == 1:  # [batch size, height, depth, in channels, width]
            output = output.permute(0, 3, 1, 4, 2)
        else:  # [batch size, height, width, in channels, depth]
            output = output.permute(0, 3, 1, 2, 4)
        return output


class AxialAttention2d(AxialAttention3d):
    """
    This class implements the axial attention operation for 2d images.
    """

    def __init__(self, in_channels: int, out_channels: int, dim: int, span: int, groups: int = 8) -> None:
        """
        Constructor method
        :param in_channels: (int) Input channels to be employed
        :param out_channels: (int) Output channels to be utilized
        :param dim: (int) Dimension attention is applied to (0 = height, 1 = width, 2 = depth)
        :param span: (int) Span of attention to be used
        :param groups: (int) Multi head attention groups to be used
        """
        # Check parameters
        assert dim in [0, 1], "Illegal argument for dimension"
        # Call super constructor
        super(AxialAttention2d, self).__index__(in_channels=in_channels, out_channels=out_channels, dim=dim, span=span,
                                                groups=groups)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        :param input: (torch.Tensor) Input tensor of the shape [batch size, in channels, h, w]
        :return: (torch.Tensor) Output tensor of the shape [batch size, out channels, h, w]
        """
        # Reshape tensor to use 3d axial-attention
        input = input.unsqueeze(dim=0)
        # Perform axial-attention
        output = super().forward(input=input)
        # Reshape output to get desired 2d tensor
        output = output.squeeze(dim=0)
        return output


class AxialAttention3dBlock(nn.Module):
    """
    This class implements the axial attention block proposed in:
    https://arxiv.org/pdf/2003.07853.pdf
    """

    def __init__(self, in_channels: int, out_channels: int, span: Union[int, Tuple[int, int, int]], groups: int = 4,
                 normalization: Type = nn.BatchNorm3d, activation: Type = nn.ReLU, downscale: bool = True,
                 dropout: float = 0.0) -> None:
        """
        Constructor method
        :param in_channels: (int) Input channels to be employed
        :param out_channels: (int) Output channels to be utilized
        :param span: (Union[int, Tuple[int, int, int]]) Spans to be used in attention layers
        :param groups: (int) Multi head attention groups to be used
        :param normalization: (Type) Type of normalization to be used
        :param activation: (Type) Type of activation to be utilized
        :param downscale: (bool) If true spatial dimensions of the output tensor are downscaled by a factor of two
        :param dropout: (float) Dropout rate to be utilized
        """
        # Call super constructor
        super(AxialAttention3dBlock, self).__init__()
        # Span to tuple
        span = span if isinstance(span, tuple) else (span, span, span)
        # Init input mapping
        self.input_mapping = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=(1, 1, 1), bias=False),
            normalization(num_features=out_channels, affine=True, track_running_stats=True),
            activation()
        )
        # Init axial attention mapping
        self.axial_attention_mapping = nn.Sequential(
            AxialAttention3d(in_channels=out_channels, out_channels=out_channels, dim=0, span=span[0], groups=groups),
            AxialAttention3d(in_channels=out_channels, out_channels=out_channels, dim=1, span=span[1], groups=groups),
            AxialAttention3d(in_channels=out_channels, out_channels=out_channels, dim=2, span=span[2], groups=groups),
        )
        # Init dropout layer
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        # Init output mapping
        self.output_mapping = nn.Sequential(
            nn.Conv3d(in_channels=out_channels, out_channels=out_channels,
                      kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=(1, 1, 1), bias=False),
            normalization(num_features=out_channels, affine=True, track_running_stats=True)
        )
        # Init residual mapping
        self.residual_mapping = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1, 1),
                                          padding=(0, 0, 0), stride=(1, 1, 1),
                                          bias=False) if in_channels != out_channels else nn.Identity()
        # Init final activation
        self.final_activation = activation()
        # Init pooling layer for downscaling the spatial dimensions
        self.pooling_layer = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)) if downscale else nn.Identity()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        :param input: (torch.Tensor) Input volume tensor of the shape [batch size, in channels, h, w, d]
        :return: (torch.Tensor) Output volume tensor of the shape [batch size, out channels, h / 2, w / 2, d / 2]
        """
        # Perform input mapping
        output = self.input_mapping(input)
        # Perform attention
        output = self.axial_attention_mapping(output)
        # Perform dropout
        output = self.dropout(output)
        # Perform output mapping
        output = self.output_mapping(self.pooling_layer(output))
        # Perform residual mapping
        output = output + self.pooling_layer(self.residual_mapping(input))
        # Perform final activation
        output = self.final_activation(output)
        return output


class AxialAttention2dBlock(nn.Module):
    """
    This class implements the axial attention block proposed in:
    https://arxiv.org/pdf/2003.07853.pdf
    """

    def __init__(self, in_channels: int, out_channels: int, span: Union[int, Tuple[int, int]], groups: int = 4,
                 normalization: Type = nn.BatchNorm2d, activation: Type = nn.ReLU, downscale: bool = True,
                 dropout: float = 0.0) -> None:
        """
        Constructor method
        :param in_channels: (int) Input channels to be employed
        :param out_channels: (int) Output channels to be utilized
        :param span: (Union[int, Tuple[int, int, int]]) Spans to be used in attention layers
        :param groups: (int) Multi head attention groups to be used
        :param normalization: (Type) Type of normalization to be used
        :param activation: (Type) Type of activation to be utilized
        :param downscale: (bool) If true spatial dimensions of the output tensor are downscaled by a factor of two
        :param dropout: (float) Dropout rate to be utilized
        """
        # Call super constructor
        super(AxialAttention2dBlock, self).__init__()
        # Span to tuple
        span = span if isinstance(span, tuple) else (span, span)
        # Init input mapping
        self.input_mapping = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), bias=False),
            normalization(num_features=out_channels, affine=True, track_running_stats=True),
            activation()
        )
        # Init axial attention mapping
        self.axial_attention_mapping = nn.Sequential(
            AxialAttention2d(in_channels=out_channels, out_channels=out_channels, dim=0, span=span[0], groups=groups),
            AxialAttention2d(in_channels=out_channels, out_channels=out_channels, dim=1, span=span[1], groups=groups),
        )
        # Init dropout layer
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        # Init output mapping
        self.output_mapping = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                      kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), bias=False),
            normalization(num_features=out_channels, affine=True, track_running_stats=True)
        )
        # Init residual mapping
        self.residual_mapping = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1),
                                          padding=(0, 0), stride=(1, 1),
                                          bias=False) if in_channels != out_channels else nn.Identity()
        # Init final activation
        self.final_activation = activation()
        # Init pooling layer for downscaling the spatial dimensions
        self.pooling_layer = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)) if downscale else nn.Identity()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        :param input: (torch.Tensor) Input volume tensor of the shape [batch size, in channels, h, w, d]
        :return: (torch.Tensor) Output volume tensor of the shape [batch size, out channels, h / 2, w / 2, d / 2]
        """
        # Perform input mapping
        output = self.input_mapping(input)
        # Perform attention
        output = self.axial_attention_mapping(output)
        # Perform dropout
        output = self.dropout(output)
        # Perform output mapping
        output = self.output_mapping(self.pooling_layer(output))
        # Perform residual mapping
        output = output + self.pooling_layer(self.residual_mapping(input))
        # Perform final activation
        output = self.final_activation(output)
        return output


