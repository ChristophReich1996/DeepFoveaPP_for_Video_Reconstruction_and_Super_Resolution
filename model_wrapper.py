import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from datetime import datetime

import lossfunction
import misc


class ModelWrapper(object):
    """
    This class wraps all modules and implements train, validation, test and inference methods
    """

    def __init__(self, generator_network: nn.Module, discriminator_network: nn.Module,
                 fft_discriminator_network: nn.Module, vgg_19: nn.Module,
                 generator_network_optimizer: torch.optim.Optimizer,
                 discriminator_optimizer: torch.optim.Optimizer,
                 fft_discriminator_optimizer: torch.optim.Optimizer, training_dataloader: DataLoader,
                 validation_dataloader: DataLoader, test_dataloader: DataLoader,
                 loss_function: nn.Module = lossfunction.AdaptiveRobustLoss(device='cuda'),
                 perceptual_loss: nn.Module = lossfunction.PerceptualLoss(),
                 generator_loss: nn.Module = lossfunction.WassersteinGeneratorLoss(),
                 discriminator_loss: nn.Module = lossfunction.WassersteinDiscriminatorLoss(), device='cuda',
                 save_data_path: str = 'saved_data') -> None:
        # Save arguments
        self.generator_network = generator_network
        self.discriminator_network = discriminator_network
        self.fft_discriminator_network = fft_discriminator_network
        self.vgg_19 = vgg_19
        self.generator_network_optimizer = generator_network_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.fft_discriminator_optimizer = fft_discriminator_optimizer
        self.training_dataloader = training_dataloader
        self.validation_dataloader = validation_dataloader
        self.test_dataloader = test_dataloader
        self.loss_function = loss_function
        self.perceptual_loss = perceptual_loss
        self.generator_loss = generator_loss
        self.discriminator_loss = discriminator_loss
        self.device = device
        self.save_data_path = save_data_path
        # Init logger
        self.logger = misc.Logger()
        # Make directories to save logs, plots and models during training
        time_and_date = str(datetime.now())
        self.path_save_models = os.path.join(save_data_path, 'models_' + time_and_date)
        if not os.path.exists(self.path_save_models):
            os.makedirs(self.path_save_models)
        self.path_save_plots = os.path.join(save_data_path, 'plots_' + time_and_date)
        if not os.path.exists(self.path_save_plots):
            os.makedirs(self.path_save_plots)
        self.path_save_metrics = os.path.join(save_data_path, 'metrics_' + time_and_date)
        if not os.path.exists(self.path_save_metrics):
            os.makedirs(self.path_save_metrics)
        # Log hyperparameter
        self.logger.hyperparameter['generator_network'] = str(generator_network)
        self.logger.hyperparameter['discriminator_network'] = str(discriminator_network)
        self.logger.hyperparameter['fft_discriminator_network'] = str(fft_discriminator_network)
        self.logger.hyperparameter['vgg_19'] = str(vgg_19)
        self.logger.hyperparameter['generator_network_optimizer'] = str(generator_network)
        self.logger.hyperparameter['generator_network'] = str(generator_network_optimizer)
        self.logger.hyperparameter['discriminator_optimizer'] = str(discriminator_optimizer)
        self.logger.hyperparameter['fft_discriminator_optimizer'] = str(fft_discriminator_optimizer)
        self.logger.hyperparameter['training_dataloader'] = str(training_dataloader)
        self.logger.hyperparameter['validation_dataloader'] = str(validation_dataloader)
        self.logger.hyperparameter['test_dataloader'] = str(test_dataloader)
        self.logger.hyperparameter['loss_function'] = str(loss_function)
        self.logger.hyperparameter['perceptual_loss'] = str(perceptual_loss)
        self.logger.hyperparameter['generator_loss'] = str(generator_loss)
        self.logger.hyperparameter['discriminator_loss'] = str(discriminator_loss)

    def train(self) -> None:
        pass

    def validate(self) -> None:
        pass

    def test(self) -> None:
        pass

    def inference(self) -> None:
        pass
