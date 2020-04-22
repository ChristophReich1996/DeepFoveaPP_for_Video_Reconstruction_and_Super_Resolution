import torch.nn as nn
import torchvision


class VGG19(nn.Module):
    '''
    This class implements a pre-trained vgg-19 classification network which returns the activations of
    the first five conv2 layers
    '''

    def __init__(self) -> None:
        '''
        Constructor method
        '''
        # Call super constructor
        super(VGG19, self).__init__()
        # Load pre-trained vgg-19 from torchvision
        self.vgg_19 = torchvision.models.vgg19()
