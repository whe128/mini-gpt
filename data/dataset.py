import torch
from torch.utils.data import Dataset

class ModelDataset(Dataset):
    """
    customized dataset class for loading data into the model
    """

    def __init__(self, token_ids, block_size):
        """
            token_ids: a list of token IDs   sequences of each sample in the dataset

        """
        self.data = token_ids
        self.block_size = block_size

    def __len__(self):
        # the training data at least need to be block_size + 1
        # at least the length need to be block_size
        # original idx: 0 ~ len(self.data) - 1
        # new      idx: 0 ~ len(self.data) - block_size - 1
        return max(0, len(self.data) - self.block_size)

    def __getitem__(self, idx):
        # y shifted by 1 position to the right
        # means that the model is trying to predict the next token given the previous tokens
        x = self.data[idx:idx + self.block_size]
        y = self.data[idx + 1:idx + self.block_size + 1]

        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(y, dtype=torch.long)
        )
