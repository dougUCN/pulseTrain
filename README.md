# pulseTrain

Pileup detection of UCNtau pulse train events. The photon probability distribution of a single UCN detection event can be visualized with `plot_discrete_pdf.py`

## Getting started

Instructions are for a Windows 10 PC with a 3070ti, Git bash and Conda

### Git Bash

Install the Git Bash utility for Windows from [Git SCM](https://git-scm.com/downloads)

(Useful note on git-credential store methods for Windows [here](https://github.com/git-ecosystem/git-credential-manager/blob/main/docs/credstores.md))

### Git LFS

The coincidences file (`coinc_4200_5441.dat`) is large and requires the use of the Git large file storage (LFS). A commit + push that involves this file counts against the 1 Gb/mo bandwidth in my account. Instructions for installing Git LFS are [here](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage).

On Ubuntu, the apt-get package manager can install Git LFS. On Windows, the Git Bash install may bundle Git LFS for you. Installation of Git LFS may be verified by running the below command in the `pulseTrain` root directory

```bash
git lfs env
```

### Clone this repository

```bash
git clone https://github.com/dougUCN/pulseTrain.git
```

### Conda

Install Conda following the instructions [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)

To get Conda to run on Git Bash, to the `~/.bashrc` configuration file add the line

```bash
source /c/Users/$USERNAME/anaconda3/etc/profile.d/conda.sh
```

with `$USERNAME` replaced appropriately, or the path modified if you chose a different installation directory for conda. Then, restart your Git Bash instance.

### CUDA install

As per Nvidia's [instructions](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html), installation of the CUDA Toolkit with Conda simply requires

```bash
conda install cuda -c nvidia
```

Verify the install with `nvcc -V`. The output for my device is

```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Tue_Jun_13_19:42:34_Pacific_Daylight_Time_2023
Cuda compilation tools, release 12.2, V12.2.91
Build cuda_12.2.r12.2/compiler.32965470_0
```

Uninstall with `conda remove cuda`

### Conda environment

Create a conda environment from the provided `environment.yml` file

```bash
conda env create -f environment.yml
conda activate pulseTrain
```

Other useful conda commands

```bash
conda install PACKAGE_NAME # Install a package
conda deactivate # Exit the environment
conda env list # List all conda environments
conda create --name ENV_NAME # Create a new conda environment carrying over dependencies from base
conda env export > environment.yml # Export active environment to a configuration file
conda remove --name ENV_NAME --all # deletes environment ENV_NAME and uninstalls associated packages
```

## Generate training, test, and validation data

```
$ python generate_training_data.py --help
usage: generate_training_data.py [-h] [-l LENGTH] [-sb STARTBUFFER] [-eb ENDBUFFER] [-ucn UCN UCN] [-p PHOTONS PHOTONS]

Generate training, validation, and test data sets via Monte Carlo

options:
  -h, --help            show this help message and exit
  -l LENGTH, --length LENGTH
                        Number of bins in the pileup window. [1 bin = 1 ns] (default: 2000)
  -sb STARTBUFFER, --startBuffer STARTBUFFER
                        Allows UCN events to occur before the pileup window by some number of bins [1 bin = 1 ns]
                        (default: 50)
  -eb ENDBUFFER, --endBuffer ENDBUFFER
                        Does NOT allow UCN events to occur before the end of the pileup window by some number of bins [1   
                        bin = 1 ns] (default: 25)
  -ucn UCN UCN, --ucn UCN UCN
                        [min, max] number of allowed UCN events per dataset (default: [0, 50])
  -p PHOTONS PHOTONS, --photons PHOTONS PHOTONS
                        [av, std_dev] Gaussian dist. of photons (integer) to generate per UCN event (default: [35, 2.5])
```

## Training the network

```
$ python train_model.py --help
usage: train_model.py [-h] [-f FILENAME] [--showModelOnly]

Train model

options:
  -h, --help            show this help message and exit
  -f FILENAME, --filename FILENAME
                        metadata filename (default: in/metadata.csv)
  --showModelOnly       Exit immediately after displaying model params (default: False)
```

## Contributing

For additional documentation and contribution guidelines see [CONTRIBUTING.md](CONTRIBUTING.md)
