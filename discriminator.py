import torch
import torch.nn as nn


class Discriminator(nn.Module):
    '''
    This class implements the discriminator network
    '''

    def __init__(self) -> None:
        '''
        Constructor method
        '''
        # Call super constructor
        super(Discriminator, self).__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass of the module
        :param input: (torch.Tensor) Input sequence of images
        :return: (torch.Tensor) True or false patch prediction
        '''
        pass


class FFTDiscriminator(nn.Module):
    '''
    This class implements the fft discriminator network
    '''

    def __init__(self) -> None:
        '''
        Constructor method
        '''
        # Call super constructor
        super(FFTDiscriminator, self).__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass of the module
        :param input: (torch.Tensor) Input sequence of images
        :return: (torch.Tensor) True or false scalar prediction
        '''
        pass
