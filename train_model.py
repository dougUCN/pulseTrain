#!/usr/bin/env python
"""
Train model
"""
# TODO Update docstring and split into ResNet1D and MSResNet files
# TODO Runtime logging
# TODO Save progress upon Ctrl C

import torch, argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
from src.models import MSResNet
from src.utils import get_project_root, get_dataset_generator
from torchinfo import summary
from sklearn.metrics import classification_report

ROOT_DIR = get_project_root()

# Parameters
LOADER_PARAMS = {
    "batch_size": 64,
    "num_workers": 4,
    "pin_memory": True,
}

# # ResNet1D
# # (n_block, downsample_gap, increasefilter_gap) = (8, 1, 2)
# # 34 layer (16*2+2): 16, 2, 4
# # 98 layer (48*2+2): 48, 6, 12
# MODEL_PARAMS = {
#     "in_channels": 1,  # Dimension of the input
#     "base_filters": 64,  # number of filters in the first several Conv layer, will double every 4 layers
#     "kernel_size": 16,  # width of kernel
#     "stride": 2,  # stride of kernel moving
#     "groups": 32, # See https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
#     "n_block": 16,  # Number of residual blocks
#     "downsample_gap": 2,
#     "increasefilter_gap": 4,
#     "n_classes": 50,  # number of labels (classes)
#     "use_do": True,  # Enable dropout
# }

# MSResNet
MODEL_PARAMS = {
    "input_channel": 1,
    "layers": [1, 1, 1, 1],
    "num_classes": 6,  # number of labels (classes)
}

MAX_EPOCHS = 10
LEARNING_RATE = 0.005  # MSResNet
# LEARNING_RATE=1e-3 #  ResNet1D
WEIGHT_DECAY = 0  # If non-zero, adds L2 penalty to loss function
# WEIGHT_DECAY = 1e-3 # ResNet1D

LOG_RATE = 1000  # Logs approximately after this many minibatches


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
    parser.add_argument(
        "--showModelOnly",
        action="store_true",
        help="Exit immediately after displaying model params",
    )
    args = parser.parse_args()

    # Parse metadata file and set up data loading
    metadata = pd.read_csv(args.filename)
    dataset_generator = get_dataset_generator(metadata, LOADER_PARAMS)

    # Get labels for validation set
    validation_labels = pd.read_csv(
        metadata.query("name == 'validation'")["label_file"].tolist()[0]
    )["label"].to_numpy()

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    # Initialize model and move to GPU
    # model = ResNet1D(**MODEL_PARAMS)
    model = MSResNet(**MODEL_PARAMS)

    model.to(device)
    model.verbose = False

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( # ResNet1D
    #     optimizer, mode="min", factor=0.1, patience=10
    # )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(  # MSResNet
        optimizer, milestones=[50, 100, 150, 200, 250, 300], gamma=0.1
    )

    # Loss function
    loss_func = torch.nn.CrossEntropyLoss()

    # Print general info
    print(f"device: {device}")
    print(LOADER_PARAMS)
    print(MODEL_PARAMS)
    print(f"learning rate: {LEARNING_RATE}")
    print(f"Training set has {len(dataset_generator['training'].dataset)} instances")
    print(
        f"Validation set has {len(dataset_generator['validation'].dataset)} instances"
    )

    # TODO: Save model output to file
    summary(model, device=device)

    if args.showModelOnly:
        exit()

    # Initialize params for tracking loss
    epochs = []
    running_losses = []
    minibatch = []

    # START EPOCH LOOP
    for epoch in tqdm(range(MAX_EPOCHS), desc="epoch"):
        running_loss = 0.0
        # Train
        model.train()
        for local_i, (local_batch, local_labels) in enumerate(
            tqdm(dataset_generator["training"], desc="Training")
        ):
            # Transfer data to GPU
            input_x, input_y = local_batch.to(device), local_labels.to(device)

            # zero parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            prediction = model(input_x)
            loss = loss_func(prediction, input_y)
            loss.backward()
            optimizer.step()

            # Log loss after some mini-batches
            running_loss += loss.item()
            if local_i % LOG_RATE == LOG_RATE - 1:
                epochs.append(epoch)
                minibatch.append(local_i)
                running_losses.append(running_loss)
                running_loss = 0.0

        scheduler.step()

        # Validation
        model.eval()
        epoch_prediction_prob = []
        with torch.no_grad():
            for local_batch, local_labels in tqdm(
                dataset_generator["validation"],
                desc="Validation",
            ):
                input_x, input_y = local_batch.to(device), local_labels.to(device)
                prediction = model(input_x)
                epoch_prediction_prob.append(
                    prediction.cpu().data.numpy()
                )  # Move tensor back to cpu

        # TODO save evaluations
        epoch_prediction_prob = np.concatenate(epoch_prediction_prob)
        epoch_prediction = np.argmax(epoch_prediction_prob, axis=1)  # Apply hardmax

        tmp_report = classification_report(
            y_true=validation_labels,
            y_pred=epoch_prediction,
        )
        print(tmp_report)
        # https://developers.google.com/machine-learning/crash-course/classification/precision-and-recall

    # END EPOCH LOOP

    # Print and save losses
    loss_df = pd.DataFrame(
        {
            "epoch": epochs,
            "minibatch": minibatch,
            "running_loss": running_losses,
        }
    )
    print(loss_df)
    loss_df.to_csv(str(ROOT_DIR / "out" / "training_loss.csv"))

    final_report = classification_report(
        y_true=validation_labels,
        y_pred=epoch_prediction,
        output_dict=True,
    )
    pd.DataFrame(final_report).transpose().to_csv(
        str(ROOT_DIR / "out" / "classification_report.csv")
    )

    # TODO Save model

    return


if __name__ == "__main__":
    main()
