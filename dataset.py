from typing import Tuple

import torch
from torch.utils.data.dataset import Dataset


class VideoDataset(Dataset):
    """
    This class implements a video dataset for super resolution
    """

    def __init__(self) -> None:
        """
        Constructor method
        """
        # Call super constructor
        super(VideoDataset, self).__init__()

    def __len__(self) -> int:
        """
        Method returns the length of the dataset
        :return: (int) Length of the dataset
        """
        pass

    def __getitem__(self, item) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Method returns one instance of the dataset for a given index
        :param item: (int) Index to return
        :return: (Tuple[torch.Tensor, torch.Tensor]) One image frame in low and high resolution
        """
        pass
