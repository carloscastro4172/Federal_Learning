import os
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from typing import Tuple
from .data_preparation import preprocess_split_dfs

class TabularDataset(Dataset):
    """Dataset tabular compatible con PyTorch"""
    def __init__(self, dataframe: pd.DataFrame, target_col: str = "target"):
        self.X = dataframe.drop(columns=[target_col]).values.astype("float32")
        self.y = dataframe[target_col].values.astype("float32")
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        return (torch.tensor(self.X[idx], dtype=torch.float32),
                torch.tensor(self.y[idx], dtype=torch.float32))

class DataManager:
    """Maneja datasets y DataLoaders (Singleton Pattern)"""
    _singleton_dm = None

    @classmethod
    def dm(cls, th: int = 0):
        if not cls._singleton_dm and th > 0:
            cls._singleton_dm = cls(th)
        return cls._singleton_dm

    def __init__(self, cutoff_th: int):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        DATA_DIR = os.path.join(BASE_DIR, "data")
        train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
        val_df   = pd.read_csv(os.path.join(DATA_DIR, "val.csv"))
        test_df  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

        train_p, val_p, test_p, feat_cols = preprocess_split_dfs(train_df, val_df, test_df)

        trainset = TabularDataset(train_p, target_col="target")
        valset   = TabularDataset(val_p,   target_col="target")
        testset  = TabularDataset(test_p,  target_col="target")

        self.trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
        self.valloader   = DataLoader(valset,   batch_size=32, shuffle=False)
        self.testloader  = DataLoader(testset,  batch_size=32, shuffle=False)

        self.cutoff_threshold = cutoff_th
        self.input_dim = len(feat_cols)

    def get_random_batch(self, is_train: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        loader = self.trainloader if is_train else self.testloader
        features, labels = next(iter(loader))
        return features, labels
