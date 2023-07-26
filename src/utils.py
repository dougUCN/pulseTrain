import torch
import pandas as pd
from pathlib import Path
from .dataset import pulse_train_dataset


def get_project_root() -> Path:
    return Path(__file__).parent.parent


def get_dataset_generator(metadata, loader_params):
    """
    Gets datasets listed by a dataframe and returns dataset_generator

    Parameters:
        metadata - Pandas df with columns: name, data_file, `label_file`

        loader_params - dictionary with kwargs for torch.utils.data.DataLoader

    Returns:
        dataset_generator - dict with keys corresponding to `name`
    """
    dataset = {}
    dataset_generator = {}

    for name, data_file in zip(
        metadata["name"].tolist(), metadata["data_file"].tolist()
    ):
        dataset[name] = pulse_train_dataset(data_file, metadata)
        if name == "train":  # Only shuffle training set
            loader_params["shuffle"] = True
        else:
            loader_params["shuffle"] = False
        dataset_generator[name] = torch.utils.data.DataLoader(
            dataset[name], **loader_params
        )
    return dataset_generator
