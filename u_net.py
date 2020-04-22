import torch
import torch.nn as nn


class RecurrentUNet(nn.Module):
    '''
    This class implements a recurrent U-Net to perform super resolution base on the DeepFovea architecture
    '''

    def __init__(self):
        '''
        Constructor method
        '''
        # Call super constructor
        super(RecurrentUNet, self).__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass
        :param input: (torch.Tensor) Input frame
        :return: (torch.Tensor) Super resolution output frame
        '''
        pass
