import random

import numpy as np
from torch.utils.data import DataLoader, Dataset


class TorchDataset(Dataset):
    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return {k: v[index] for k, v in self.x.items()}, self.y[index]

    def __len__(self):
        return len(self.y)


class DataGenerator(object):
    def __init__(self, x, y):
        super().__init__()
        self.dataset = TorchDataset(x, y)

    def generate_dataloader(self, x_val=None, y_val=None, x_test=None, y_test=None, batch_size=16, num_workers=0):
        train_dataset = self.dataset
        val_dataset = TorchDataset(x_val, y_val)
        test_dataset = TorchDataset(x_test, y_test)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return train_dataloader, val_dataloader, test_dataloader


def get_auto_embedding_dim(num_classes):
    return int(np.floor(6 * np.power(num_classes, 0.25)))


def pad_sequences(sequences, maxlen=None, dtype="int32", padding="pre", truncating="pre", value=0.0):
    if maxlen is None:
        maxlen = max(len(item) for item in sequences)
    x = np.full((len(sequences), maxlen), value, dtype=dtype)
    for idx, seq in enumerate(sequences):
        if not len(seq):
            continue
        if truncating == "pre":
            trunc = seq[-maxlen:]
        elif truncating == "post":
            trunc = seq[:maxlen]
        else:
            raise ValueError(f"Unsupported truncating mode: {truncating}")
        trunc = np.asarray(trunc, dtype=dtype)
        if padding == "post":
            x[idx, : len(trunc)] = trunc
        elif padding == "pre":
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError(f"Unsupported padding mode: {padding}")
    return x


def neg_sample(click_hist, item_size):
    neg = random.randint(1, item_size)
    while neg in click_hist:
        neg = random.randint(1, item_size)
    return neg