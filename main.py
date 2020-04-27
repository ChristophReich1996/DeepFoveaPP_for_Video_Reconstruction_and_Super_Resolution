import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import torch
from torch.utils.data import DataLoader

from u_net import RecurrentUNet
from discriminator import Discriminator, FFTDiscriminator
from vgg_19 import VGG19
from model_wrapper import ModelWrapper
from dataset import REDS, REDSFovea
from lossfunction import AdaptiveRobustLoss

if __name__ == '__main__':
    # Init networks
    generator_network = RecurrentUNet()
    discriminator_network = Discriminator()
    fft_discriminator_network = FFTDiscriminator()
    vgg_19 = VGG19()
    # Init adaptive loss
    loss_function = AdaptiveRobustLoss(device='cuda:0', num_of_dimension=3 * 6 * 1024 * 768)
    # Init optimizers
    generator_network_optimizer = torch.optim.Adam(
        list(generator_network.parameters()) + list(loss_function.parameters()), lr=1e-4)
    discriminator_network_optimizer = torch.optim.Adam(discriminator_network.parameters(), lr=2e-4)
    fft_discriminator_network_optimizer = torch.optim.Adam(fft_discriminator_network.parameters(), lr=2e-4)
    # Init model wrapper
    model_wrapper = ModelWrapper(generator_network=generator_network,
                                 discriminator_network=discriminator_network,
                                 fft_discriminator_network=fft_discriminator_network,
                                 vgg_19=vgg_19,
                                 generator_network_optimizer=generator_network_optimizer,
                                 discriminator_network_optimizer=discriminator_network_optimizer,
                                 fft_discriminator_network_optimizer=fft_discriminator_network_optimizer,
                                 loss_function=loss_function,
                                 training_dataloader=DataLoader(
                                     REDSFovea(path='/home/creich/REDS/train/train_sharp'), batch_size=1, shuffle=False),
                                 validation_dataloader=DataLoader(
                                     REDSFovea(path='/home/creich/REDS/val/val_sharp'), batch_size=1, shuffle=False),
                                 test_dataloader=None)
    model_wrapper.validate()
    # Perform training
    model_wrapper.train(epochs=10)
