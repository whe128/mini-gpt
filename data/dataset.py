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

class SFTDataset(Dataset):
    """
    customized dataset class for loading data into the model
    """

    def __init__(self, inputs, outputs, block_size, eos_token_id, pad_token_id = -100):
        """
            prompts, labels: a list of token IDs  sequences of each sample in the dataset
            [
                [token_id_1, token_id_2, ..., token_id_n],
                [token_id_1, token_id_2, ..., token_id_m],
                ...
            ]

        """
        self.inputs = inputs
        self.outputs = outputs
        self.block_size = block_size
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

    def __len__(self):
        # one sample is a list of token IDs, so the length of the dataset is the number of samples
        return len(self.inputs)

    def __getitem__(self, idx):
        x = self.inputs[idx]
        y = self.outputs[idx]

        # truncate or pad the sequences to the block_size
        if len(x) > self.block_size:
            x = x[:self.block_size]
            y = y[:self.block_size]

            y[-1] = self.eos_token_id           # make sure the last token is eos
        else:
            x = x + [self.eos_token_id] * (self.block_size - len(x))  # pad with eos
            y = y + [self.pad_token_id] * (self.block_size - len(y))  # pad with -100

        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(y, dtype=torch.long)
        )
