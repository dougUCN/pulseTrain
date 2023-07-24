#!/usr/bin/env python
"""
Train model
"""

import torch, argparse
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from src.dataset import pulse_train_dataset


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
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

    # Parameters
    params = {"batch_size": 64, "shuffle": True, "num_workers": 4, "pin_memory": True}
    max_epochs = 50

    # Parse metadata file and set up data loading
    dataset = {}
    dataset_generator = {}
    metadata = pd.read_csv(args.filename)
    for name, data_file in zip(
        metadata["name"].tolist(), metadata["data_file"].tolist()
    ):
        dataset[name] = pulse_train_dataset(data_file, args.filename)
        dataset_generator = torch.utils.data.DataLoader(dataset[name], **params)

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    # Loop over epochs
    for epoch in tqdm(range(max_epochs)):
        # Train
        for local_batch, local_labels in dataset_generator["training"]:
            # Transfer to GPU
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)

            # Model computations
            # [...]

        # Validation
        with torch.set_grad_enabled(False):
            for local_batch, local_labels in dataset_generator["validation"]:
                # Transfer to GPU
                local_batch, local_labels = local_batch.to(device), local_labels.to(
                    device
                )

                # Model computations
                # [...]

    return


if __name__ == "__main__":
    main()
