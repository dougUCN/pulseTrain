#!/usr/bin/env python
"""
Contains r/w operations and classes related to dataset manipulation

MemmapDataset from https://saturncloud.io/blog/efficient-way-of-using-numpy-memmap-when-training-neural-network-with-pytorch/
"""
import torch
import numpy as np
import pandas as pd
from pathlib import Path


class pulse_train_dataset(torch.utils.data.Dataset):
    def __init__(self, memmap_file, metadata_file):
        """
        Parameters:
            memmap_file - numpy memmap
            metadata_file - csv that contains memmap dimensions and points to label file
        """
        data_absolute_path = str(Path(memmap_file).resolve())
        metadata = pd.read_csv(str(Path(metadata_file).resolve())).query(
            "data_file == @data_absolute_path"
        )
        self.shape = (
            metadata["sample_number"].tolist()[0],
            metadata["sample_length"].tolist()[0],
        )
        self.labels = pd.read_csv(
            str(Path(metadata["label_file"].tolist()[0]).resolve())
        )
        self.data = np.memmap(
            data_absolute_path, dtype="int32", mode="r", shape=self.shape
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Parameters:
            idx - integer ID of the data set

        Returns:
            x - torch.tensor
            label - Labels of the dataset
        """
        x = torch.from_numpy(self.data[idx, :].copy())  # avoid a pytorch warning
        label = self.labels.iloc[idx]["label"]
        return x, label


def write_memmap_to_file(outfile, data, verbose=True):
    """
    Parameters:
        outfile - name of output file
        data - 2D numpy array of ints
    """
    out = np.memmap(outfile, dtype="int32", mode="w+", shape=data.shape)
    if verbose:
        print(data)
    out[:] = data[:]
    out.flush()


def read_memmap(infile, shape):
    """
    Parameters:
        infile - name of memmap file
        shape - tuple (x, y)

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
    """Test PyTorch dataloader and determine optimal number of workers"""
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

        # Sample first 3 in dataset
        print(name)
        for i, sample in enumerate(dataset[name]):
            print(f"{sample} {sample[0].size()}")

            if i == 2:
                print("...")
                break

    # Runtime batching test with the dataset with the most samples to see optimal worker count
    # https://chtalhaanwar.medium.com/pytorch-num-workers-a-tip-for-speedy-training-ed127d825db7#:~:text=Num_workers%20tells%20the%20data%20loader,the%20GPU%20has%20to%20wait.
    from time import time
    import multiprocessing as mp

    most_samples = metadata[
        metadata["sample_number"] == metadata["sample_number"].max()
    ]
    # dataloader = DataLoader(dataset[row["name"]],shuffle=True,batch_size=10,pin_memory=True)
    # for i_batch, sample_batched in enumerate(dataloader):
    #     print(i_batch, sample_batched['x'].size(),
    #         sample_batched['landmarks'].size())

    return


if __name__ == "__main__":
    main()
