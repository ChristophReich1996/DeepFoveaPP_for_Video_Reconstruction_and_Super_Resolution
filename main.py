import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0, 2'

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from u_net import RecurrentUNet
from discriminator import Discriminator, FFTDiscriminator
from vgg_19 import VGG19
from pwc_net import PWCNet
from model_wrapper import ModelWrapper
from dataset import REDS, REDSFovea, REDSParallel, REDSFoveaParallel, reds_parallel_collate_fn
from lossfunction import AdaptiveRobustLoss
from resample.resample2d import Resample2d

if __name__ == '__main__':
    # Init networks
    generator_network = nn.DataParallel(RecurrentUNet().cuda())
    discriminator_network = nn.DataParallel(Discriminator().cuda())
    fft_discriminator_network = nn.DataParallel(FFTDiscriminator().cuda())
    vgg_19 = nn.DataParallel(VGG19().cuda())
    pwc_net = nn.DataParallel(PWCNet().cuda())
    resample = nn.DataParallel(Resample2d().cuda())
    # Init adaptive loss
    loss_function = nn.L1Loss()
    # Init optimizers
    generator_network_optimizer = torch.optim.Adam(
        list(generator_network.parameters()) + list(loss_function.parameters()), lr=3e-4, betas=(0.1, 0.95))
    discriminator_network_optimizer = torch.optim.Adam(discriminator_network.parameters(), lr=1e-4, betas=(0.1, 0.95))
    fft_discriminator_network_optimizer = torch.optim.Adam(fft_discriminator_network.parameters(), lr=1e-4,
                                                           betas=(0.1, 0.95))
    # Init model wrapper
    model_wrapper = ModelWrapper(generator_network=generator_network,
                                 discriminator_network=discriminator_network,
                                 fft_discriminator_network=fft_discriminator_network,
                                 vgg_19=vgg_19,
                                 pwc_net=pwc_net,
                                 resample=resample,
                                 generator_network_optimizer=generator_network_optimizer,
                                 discriminator_network_optimizer=discriminator_network_optimizer,
                                 fft_discriminator_network_optimizer=fft_discriminator_network_optimizer,
                                 loss_function=loss_function,
                                 training_dataloader=DataLoader(
                                     REDSFoveaParallel(path='/home/creich/REDS/train/train_sharp', number_of_gpus=2),
                                     batch_size=2, shuffle=False, num_workers=1, collate_fn=reds_parallel_collate_fn),
                                 validation_dataloader=DataLoader(
                                     REDSFovea(path='/home/creich/REDS/val/val_sharp'), batch_size=1, shuffle=False,
                                     num_workers=1),
                                 test_dataloader=None)
    # Perform training
    model_wrapper.train(epochs=20)
