# importing models, making pipelines...
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer


class BeerDataset(Dataset):
    """
    A dataset implements 2 functions
        - __len__  (returns the number of samples in our dataset)
        - __getitem__ (returns a sample from the dataset at the given index idx)
    """

    def __init__(self, dataset_parameters, dataset: list):
        super().__init__()
        self.dataset_parameters = dataset_parameters
        self.data = dataset

    def load_data(self):
        pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
