#!/usr/bin/env python
"""
Train model
"""

import torch, argparse
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from src.dataset import pulse_train_dataset

# Parameters
LOADER_PARAMS = {
    "batch_size": 64,
    "shuffle": True,
    "num_workers": 4,
    "pin_memory": True,
}
MAX_EPOCHS = 25
LEARNING_RATE = 0.01


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-f",
        "--filename",
        type=str,
        default="in/metadata.csv",
        help="metadata filename",
    )
    args = parser.parse_args()

    # Parse metadata file and set up data loading
    dataset = {}
    dataset_generator = {}
    metadata = pd.read_csv(args.filename)
    for name, data_file in zip(
        metadata["name"].tolist(), metadata["data_file"].tolist()
    ):
        dataset[name] = pulse_train_dataset(data_file, args.filename)
        dataset_generator[name] = torch.utils.data.DataLoader(
            dataset[name], **LOADER_PARAMS
        )

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    # Initialize model

    # Optimizer

    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    # Print general info
    print(f"device: {device}")
    print(LOADER_PARAMS)
    print(f"learning rate: {LEARNING_RATE}")
    print(f"Training set has {len(dataset_generator['training'])} instances")
    print(f"Validation set has {len(dataset_generator['validation'])} instances")

    # TRAINING LOOP
    for epoch in tqdm(range(MAX_EPOCHS)):
        for local_batch, local_labels in dataset_generator["training"]:
            # Transfer to GPU
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)

            # Zero gradients for every batch
            optimizer.zero_grad()

            # Forward pass.

            # Calculate the loss and accuracy

            # Backpropagation

            # Update the weights.
            optimizer.step()

        # Calculate loss and accuracy for the complete epoch.

        # Validation
        with torch.set_grad_enabled(False):
            for local_batch, local_labels in dataset_generator["validation"]:
                # Transfer to GPU
                local_batch, local_labels = local_batch.to(device), local_labels.to(
                    device
                )

                # Model computations
                # [...]

    # END TRAINING LOOP

    # TODO Save model

    return


if __name__ == "__main__":
    main()
