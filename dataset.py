from typing import Tuple

import torch
from torch.utils.data.dataset import Dataset
from torch.nn.functional import interpolate
import torchvision
from PIL import Image
import os


class REDS(Dataset):
    """
    This class implements a video dataset for super resolution
    """

    def __init__(self, path: str = '/home/creich/REDS/train/train_sharp', number_of_frames: int = 12,
                 overlapping_frames: int = 2, frame_format='png',
                 transformations: torchvision.transforms.Compose = torchvision.transforms.Compose(
                     [torchvision.transforms.CenterCrop((1024, 768)),
                      torchvision.transforms.ToTensor()])) -> None:
        """
        Constructor method
        :param path: (str) Path to data
        :param number_of_frames: (int) Number of frames in one dataset element
        :param overlapping_frames: (int) Number of overlapping frames of two consecutive dataset elements
        :param frame_format: (str) Frame format to detect frames in path
        """
        # Call super constructor
        super(REDS, self).__init__()
        # Save arguments
        self.number_of_frames = number_of_frames
        self.transformations = transformations
        # Init previously loaded frames
        self.previously_loaded_frames = None
        # Init list to store all path to frames
        self.data_path = []
        # Get all objects in path an search for video folders
        for video in sorted(os.listdir(path=path)):
            # Case that object in path is a folder
            if os.path.isdir(os.path.join(path, video)):
                # Init frame counter
                frame_counter = 0
                # Init frame index
                frame_index = 0
                # Iterate over all frames in video folder
                while frame_index < len(os.listdir(path=os.path.join(path, video))):
                    # Get current frame name
                    current_frame = sorted(os.listdir(path=os.path.join(path, video)))[frame_index]
                    # Check object is a frame of the desired format
                    if frame_format in current_frame:
                        # Add new list to data path in case of a new frame sequence
                        if frame_counter == 0:
                            self.data_path.append([])
                        # Add frame to last data path under list
                        self.data_path[-1].append(os.path.join(path, video, current_frame))
                        # Increment frame counter
                        frame_counter += 1
                        # Reset frame counter if number of frames in one element are reached
                        if frame_counter == number_of_frames:
                            frame_counter = 0
                            # Decrement frame index by the number of overlapping frames
                            frame_index -= overlapping_frames
                        # Increment frame index
                        frame_index += 1
                # Remove last list element of data_path if number of frames is not matched
                if len(self.data_path[-1]) != number_of_frames:
                    del self.data_path[-1]

    def __len__(self) -> int:
        """
        Method returns the length of the dataset
        :return: (int) Length of the dataset
        """
        return len(self.data_path)

    @torch.no_grad()
    def __getitem__(self, item) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        """
        Method returns one instance of the dataset for a given index
        :param item: (int) Index to return
        :return: (Tuple[torch.Tensor, torch.Tensor]) One image frame in low and high resolution
        """
        # Check if current frame sequence is a new video sequence
        if self.previously_loaded_frames is None or self.previously_loaded_frames[0].split('/')[-2] != \
                self.data_path[item][0].split('/')[-2]:
            new_video = True
        else:
            new_video = False
        # Set current data path to previously loaded frames
        self.previously_loaded_frames = self.data_path[item]
        # Load frames
        frames = []
        for frame in self.data_path[item]:
            # Load images as PIL image, apply transformation and append to list of frames
            frames.append(self.transformations(Image.open(frame)))
        # Concatenate frames to tensor of shape (3 * number of frames, height, width)
        frames = torch.cat(frames, dim=0)
        # Normalize whole sequence of frames
        frames = frames.sub_(frames.mean()).div_(frames.std())
        # Downsampled frames
        frames_downsampled = interpolate(frames[None], scale_factor=0.25, mode='bilinear', align_corners=False)[0]
        # Returns frames and a downscaled version as input
        return frames_downsampled, frames, new_video


class PseudoDataset(Dataset):
    """
    This class implements a pseudo dataset to test the implemented architecture
    """

    def __init__(self, length: int = 100) -> None:
        """
        Constructor method
        :param length: (int) Pseudo dataset length
        """
        # Call super constructor
        super(PseudoDataset, self).__init__()
        # Save length parameter
        self.length = length

    def __len__(self) -> int:
        """
        Method to get length of the dataset
        :return: (int) Length
        """
        return self.length

    def __getitem__(self, item) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Method returns a tensor of the shape (rgb * 16 frames, height, width) and the corresponding high res. label
        :param item: (int) Index
        :return: (torch.Tensor) Pseudo image sequence and corresponding label
        """
        if item >= len(self):
            raise IndexError
        return torch.ones([3 * 16, 64, 64], dtype=torch.float), torch.ones([3 * 16, 256, 256], dtype=torch.float)
