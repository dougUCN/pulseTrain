#!/usr/bin/env python
"""
Contains r/w operations and classes related to dataset manipulation
"""
import torch
import numpy as np
import pandas as pd
from pathlib import Path

DATA_FORMAT = "float32"


class pulse_train_dataset(torch.utils.data.Dataset):
    """
    MemmapDataset
    """

    def __init__(self, memmap_file, metadata_df, transform=None, target_transform=None):
        """
        Parameters:
            memmap_file - (str) numpy memmap filename

            metadata_df - (pandas df) contains memmap dimensions and points to label file
        """
        data_absolute_path = str(Path(memmap_file).resolve())
        metadata = metadata_df.query("data_file == @data_absolute_path")
        self.shape = (
            metadata["sample_number"].tolist()[0],
            metadata["n_channels"].tolist()[0],
            metadata["sample_length"].tolist()[0],
        )
        self.labels = pd.read_csv(
            str(Path(metadata["label_file"].tolist()[0]).resolve())
        )
        self.data = np.memmap(
            data_absolute_path, dtype=DATA_FORMAT, mode="r", shape=self.shape
        )
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Parameters:
            idx - integer ID of the data set

        Returns:
            x - np.array

            label - Labels of the dataset
        """
        x = self.data[idx, :]
        label = self.labels.iloc[idx]["label"]
        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            label = self.target_transform(label)
        return x, label


def write_memmap_to_file(outfile, data, verbose=False):
    """
    Parameters:
        outfile - name of output file

        data - numpy array of ints
    """
    out = np.memmap(outfile, dtype=DATA_FORMAT, mode="w+", shape=data.shape)
    if verbose:
        print(data)
    out[:] = data[:]
    out.flush()


def read_memmap(infile, shape):
    """
    Parameters:
        infile - name of memmap file

        shape - tuple

    Returns:
        np.memmap
    """
    data_absolute_path = str(Path(infile).resolve())
    return np.memmap(
        data_absolute_path,
        dtype="int32",
        mode="r",
        shape=shape,
    )


def main():
    """Test PyTorch dataloader"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Test PyTorch dataloader",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-f",
        "--filename",
        type=str,
        required=True,
        help="metadata filename",
    )
    args = parser.parse_args()

    dataset = {}
    metadata = pd.read_csv(args.filename)
    for name, data_file in zip(
        metadata["name"].tolist(), metadata["data_file"].tolist()
    ):
        dataset[name] = pulse_train_dataset(data_file, args.filename)

        # Sample first 3 in dataset and print
        print(name)
        for i, sample in enumerate(dataset[name]):
            print(f"{sample} {sample[0].size}")

            if i == 2:
                print("...\n")
                break

    most_samples = metadata[  # get dataset name with most samples
        metadata["sample_number"] == metadata["sample_number"].max()
    ]["name"].tolist()[0]

    # Following example from
    # https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel

    # Parameters
    params = {"batch_size": 64, "shuffle": True, "num_workers": 4, "pin_memory": True}
    max_epochs = 1

    print("testing DataLoader")
    print(params)
    dataset_generator = torch.utils.data.DataLoader(dataset[most_samples], **params)

    for epoch in range(max_epochs):
        for local_batch, local_labels in dataset_generator:
            print(local_batch[0], local_labels[0])

    return


if __name__ == "__main__":
    main()
