import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from datetime import datetime
from tqdm import tqdm

import lossfunction
import misc


class ModelWrapper(object):
    """
    This class wraps all modules and implements train, validation, test and inference methods
    """

    def __init__(self, generator_network: nn.Module, discriminator_network: nn.Module,
                 fft_discriminator_network: nn.Module, vgg_19: nn.Module,
                 generator_network_optimizer: torch.optim.Optimizer,
                 discriminator_network_optimizer: torch.optim.Optimizer,
                 fft_discriminator_network_optimizer: torch.optim.Optimizer, training_dataloader: DataLoader,
                 validation_dataloader: DataLoader, test_dataloader: DataLoader,
                 loss_function: nn.Module = lossfunction.AdaptiveRobustLoss(device='cuda:0',
                                                                            num_of_dimension=3 * 16 * 128 ** 2),
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
        self.discriminator_network_optimizer = discriminator_network_optimizer
        self.fft_discriminator_network_optimizer = fft_discriminator_network_optimizer
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
        # Not compatible with windows!!!
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
        self.logger.hyperparameter['discriminator_network_optimizer'] = str(discriminator_network_optimizer)
        self.logger.hyperparameter['fft_discriminator_network_optimizer'] = str(fft_discriminator_network_optimizer)
        self.logger.hyperparameter['training_dataloader'] = str(training_dataloader)
        self.logger.hyperparameter['validation_dataloader'] = str(validation_dataloader)
        self.logger.hyperparameter['test_dataloader'] = str(test_dataloader)
        self.logger.hyperparameter['loss_function'] = str(loss_function)
        self.logger.hyperparameter['perceptual_loss'] = str(perceptual_loss)
        self.logger.hyperparameter['generator_loss'] = str(generator_loss)
        self.logger.hyperparameter['discriminator_loss'] = str(discriminator_loss)

    def train(self, epochs: int = 1, save_models_after_n_epochs: int = 4, validate_after_n_epochs: int = 4) -> None:
        """
        Train method
        Note: GPU memory issues if all losses are computed at one. Solution: Calc losses independently. Drawback:
        Multiple forward passes are needed -> Slower training. Additionally gradients are not as smooth.
        :param epochs: (int) Number of epochs to perform
        """
        # Model into training mode
        self.generator_network.train()
        self.discriminator_network.train()
        self.fft_discriminator_network.train()
        # Vgg into eval mode
        self.vgg_19.eval()
        # Models to device
        self.generator_network.to(self.device)
        self.discriminator_network.to(self.device)
        self.fft_discriminator_network.to(self.device)
        self.vgg_19.to(self.device)
        # Init progress bar
        self.progress_bar = tqdm(total=epochs * len(self.training_dataloader.dataset))
        # Main loop
        for epoch in range(epochs):
            for input, label in self.training_dataloader:
                # Update progress bar
                self.progress_bar.update(n=input.shape[0])
                # Reset gradients of networks
                self.generator_network.zero_grad()
                self.discriminator_network.zero_grad()
                self.fft_discriminator_network.zero_grad()
                self.vgg_19.zero_grad()
                # Data to device
                input = input.to(self.device)
                label = label.to(self.device)
                ############# Supervised training (+ perceptrual training) #############
                # Make prediction
                prediction = self.generator_network(input)[-1]
                # Reshape prediction and label for vgg19
                prediction_reshaped_4d = prediction.reshape(prediction.shape[0] * (prediction.shape[1] // 3), 3,
                                                            prediction.shape[2], prediction.shape[3])
                label_reshaped_4d = label.reshape(label.shape[0] * (label.shape[1] // 3), 3, label.shape[2],
                                                  label.shape[3])
                # Call supervised loss
                loss_supervised = self.loss_function(prediction, label) \
                                  + self.perceptual_loss(self.vgg_19(prediction_reshaped_4d),
                                                         self.vgg_19(label_reshaped_4d))
                # Calc gradients
                loss_supervised.backward()
                # Optimize generator
                self.generator_network_optimizer.step()
                # Reset gradients of generator network and vgg 19
                self.generator_network.zero_grad()
                self.vgg_19.zero_grad()
                ############# Adversarial training #############
                # Make prediction
                prediction = self.generator_network(input)[-1]
                # Reshape label and prediction for discriminator
                prediction_reshaped_5d = prediction.reshape(prediction.shape[0], 3, prediction.shape[1] // 3,
                                                            prediction.shape[2], prediction.shape[3])
                label_reshaped_5d = label.reshape(label.shape[0], 3, label.shape[1] // 3, label.shape[2],
                                                  label.shape[3])
                # Calc discriminator loss
                loss_discriminator = self.discriminator_loss(self.discriminator_network(label_reshaped_5d),
                                                             self.discriminator_network(prediction_reshaped_5d))
                # Calc gradients and retain graph of generator gradients
                loss_discriminator.backward(retain_graph=True)
                # Calc generator loss
                loss_generator = self.generator_loss(self.discriminator_network(prediction_reshaped_5d))
                # Calc gradients
                loss_generator.backward()
                # Optimize generator and discriminator
                self.generator_network_optimizer.step()
                self.discriminator_network_optimizer.step()
                # Reset gradients of generator and discriminator
                self.generator_network.zero_grad()
                self.discriminator_network.zero_grad()
                ############# Adversarial training (FFT) #############
                # Make prediction
                prediction = self.generator_network(input)[-1]
                # Reshape label and prediction for discriminator
                prediction_reshaped_5d = prediction.reshape(prediction.shape[0], 3, prediction.shape[1] // 3,
                                                            prediction.shape[2], prediction.shape[3])
                label_reshaped_5d = label.reshape(label.shape[0], 3, label.shape[1] // 3, label.shape[2],
                                                  label.shape[3])
                # Calc discriminator loss
                loss_fft_discriminator = self.discriminator_loss(self.fft_discriminator_network(label_reshaped_5d),
                                                                 self.fft_discriminator_network(prediction_reshaped_5d))
                # Calc gradients and retain graph of generator gradients
                loss_fft_discriminator.backward(retain_graph=True)
                # Calc generator loss
                loss_fft_generator = self.generator_loss(self.fft_discriminator_network(prediction_reshaped_5d))
                # Calc gradients
                loss_fft_generator.backward()
                # Optimize generator and discriminator
                self.generator_network_optimizer.step()
                self.fft_discriminator_network_optimizer.step()
                # Reset gradients of generator and discriminator
                self.generator_network.zero_grad()
                self.fft_discriminator_network.zero_grad()
                # Update progress bar
                self.progress_bar.set_description(
                    'SV Loss={:.4f}, Adv. G. Loss={:.4f}, Adv. D. Loss={:.4f}, Adv. FFTG. Loss={:.4f}, Adv. FFTD. Loss={:.4f}'
                        .format(loss_supervised.item(), loss_generator.item(), loss_discriminator.item(),
                                loss_fft_generator.item(), loss_fft_discriminator.item()))
                # Log losses
                self.logger.log(metric_name='training_iteration', value=self.progress_bar.n)
                self.logger.log(metric_name='epoch', value=epoch)
                self.logger.log(metric_name='loss_supervised', value=loss_supervised)
                self.logger.log(metric_name='loss_generator', value=loss_generator)
                self.logger.log(metric_name='loss_discriminator', value=loss_discriminator)
                self.logger.log(metric_name='loss_fft_generator', value=loss_fft_generator)
                self.logger.log(metric_name='loss_fft_discriminator', value=loss_fft_discriminator)
            # Save models and optimizer
            if epoch % save_models_after_n_epochs == 0:
                # Save models
                torch.save(self.generator_network.state_dict(), 'generator_network_{}.pt'.format(epoch))
                torch.save(self.discriminator_network.state_dict(), 'discriminator_network_{}.pt'.format(epoch))
                torch.save(self.fft_discriminator_network.state_dict(), 'fft_discriminator_network_{}.pt'.format(epoch))
                # Save optimizers
                torch.save(self.generator_network_optimizer, 'generator_network_optimizer_{}.pt'.format(epoch))
                torch.save(self.discriminator_network_optimizer, 'discriminator_network_optimizer_{}.pt'.format(epoch))
                torch.save(self.fft_discriminator_network_optimizer,
                           'fft_discriminator_network_optimizer_{}.pt'.format(epoch))
            if epoch % validate_after_n_epochs == 0:
                # Validation
                pass
            # Save logs
            self.logger.save_metrics(self.path_save_metrics)
        # Close progress bar
        self.progress_bar.close()

    def validate(self) -> None:
        pass

    def test(self) -> None:
        pass

    def inference(self) -> None:
        pass
