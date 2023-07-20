#!/usr/bin/env python
"""
Generate training, validation, and test data sets via Monte Carlo
"""

import argparse, tqdm
import scipy.stats as st
import numpy as np
from tqdm import tqdm
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'

from src import globals, dataset

ROOT_DIR = globals.get_project_root()

DATASETS = {  # Dataset name : number of events
    "training": 700,
    "validation": 200,
    "test": 100,
}

FILE_METADATA = "metadata.csv"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-l",
        "--length",
        type=int,
        help="Number of bins in the pileup window. [1 bin = 1 ns]",
        default=2000,
    )
    parser.add_argument(
        "-sb",
        "--startBuffer",
        type=int,
        help=(
            "Allows UCN events to occur before the "
            "pileup window by some number of bins [1 bin = 1 ns]"
        ),
        default=50,
    )
    parser.add_argument(
        "-eb",
        "--endBuffer",
        type=int,
        help=(
            "Does NOT allow UCN events to occur before the end "
            "of the pileup window by some number of bins [1 bin = 1 ns]"
        ),
        default=25,
    )
    parser.add_argument(
        "-ucn",
        "--ucn",
        type=int,
        help="[min, max] number of allowed UCN events per dataset",
        default=[0, 50],
        nargs=2,
    )
    parser.add_argument(
        "-p",
        "--photons",
        type=int,
        help="[av, std_dev] Gaussian dist. of photons (integer) to generate per UCN event",
        default=[35, 2.5],
        nargs=2,
    )

    args = parser.parse_args()
    print(args)

    # Confirm overwrite if files exist
    for filename, num_events in DATASETS.items():
        existing_files = []
        if (ROOT_DIR / "in" / (filename + ".npy")).is_file():
            existing_files.append(str(ROOT_DIR / "in" / (filename + ".npy")))
        if (ROOT_DIR / "in" / (filename + ".csv")).is_file():
            existing_files.append(str(ROOT_DIR / "in" / (filename + ".csv")))
    if existing_files:
        print(f"WARNING -- EXISTING FILE(S): {existing_files}\nwill be overwritten!")
        if input("Continue? (y/n):").lower().startswith("n"):
            exit()

    df = pd.read_table(str(ROOT_DIR / "in" / "coinc_4200_5441.dat"))
    df = df.rename(columns={df.columns[0]: "cumulative"})
    cdf = df["cumulative"].to_numpy()  # cumulative distribution function
    pdf = (
        cdf[1:] - cdf[:-1]
    )  # probability distribution function arises from differential
    time = np.arange(len(cdf) - 1) * 800e-12  # seconds

    # Generate probability distribution from histogram
    # Each bin in default distribution is 0.8 ns
    pulse_train_dist = st.rv_histogram((pdf[0:2400], time[0:2401]))
    rng = np.random.default_rng()
    outfile_name = []
    dimensions = []

    for filename, num_events in DATASETS.items():
        data, labels = generate_data(args, num_events, pulse_train_dist, rng)
        outfile_name.append(str((ROOT_DIR / "in" / filename).resolve()) + ".npy")
        dimensions.append(data.shape)
        dataset.write_memmset_to_file(
            outfile=outfile_name[-1],
            data=data,
        )
        labels.to_csv(str(ROOT_DIR / "in" / filename) + ".csv")

    # Save memmset metadata
    metadata = pd.DataFrame(dimensions, columns=["sample_number", "sample_length"])
    metadata["filename"] = outfile_name
    metadata.to_csv(str(ROOT_DIR / "in" / FILE_METADATA))

    return


def generate_data(
    args,
    num_samples,
    pulse_train_dist,
    rng,
):
    """
    Parameters:
        args - from argparse
        num_samples - number of total samples to generate
        pulse_train_dist - scipy.stats.rv_histogram
        rng - numpy rng generator

    Returns:
        data - np.array of size num_samples x args.length
        labels - pandas df with an id (for each samples) + number of UCN events
    """
    idx = []
    num_events = []
    data = np.zeros((num_samples, args.length))

    for i, d in enumerate(tqdm(data)):
        # Iterate through each set and randomize some number of UCN events
        idx.append(i)
        num_events.append(
            rng.integers(low=args.ucn[0], high=args.ucn[1], endpoint=True)
        )

        for j in range(num_events[-1]):
            # Determine number of photons per UCN event
            num_photons = int(rng.normal(loc=args.photons[0], scale=args.photons[1]))
            # Re-bin pulse train to 1 ns bins. Generate 1 UCN event
            hist, _ = np.histogram(
                pulse_train_dist.rvs(size=num_photons), range=[0, 2e-6], bins=2000
            )

            # Randomize start time of the pulse train
            start_time = rng.integers(
                low=-args.startBuffer,
                high=args.length - args.endBuffer,
                endpoint=True,
            )
            if start_time >= 0:
                d += np.pad(hist, (start_time, 0))[: args.length]
            else:
                d += np.pad(hist, (0, abs(start_time)))[-args.length :]

    return data, pd.DataFrame({"idx": idx, "label": num_events}).set_index("idx")


if __name__ == "__main__":
    main()
