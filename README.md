# pulseTrain

## Getting started

Instructions are for a Windows 10 PC with a 3070ti, Git bash and Conda

### Git Bash

Install the Git Bash utility for Windows from [Git SCM](https://git-scm.com/downloads)

(Useful note on git-credential store methods for Windows [here](https://github.com/git-ecosystem/git-credential-manager/blob/main/docs/credstores.md))

### Git LFS

The coincidences file (`coinc_4200_5441.dat`) is large and requires the use of the Git large file storage (LFS). A commit + push that involves this file counts against the 1 Gb/mo bandwidth in my account. Instructions for installing Git LFS are [here](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage).

On Ubuntu, the apt-get package manager can install Git LFS. On Windows, the Git Bash install may bundle Git LFS for you. Installation of Git LFS may be verified by running the below command in the `pulseTrain` root directory

```
git lfs env
```

### Clone this repository

```
git clone https://github.com/dougUCN/pulseTrain.git
```

### Conda

Install Conda following the instructions [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)

To get Conda to run on Git Bash, to the `~/.bashrc` configuration file add the line

```
source /c/Users/$USERNAME/anaconda3/etc/profile.d/conda.sh
```

with `$USERNAME` replaced appropriately, or the path modified if you chose a different installation directory for conda. Then, restart your Git Bash instance.

### CUDA install

As per Nvidia's [instructions](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html), installation of the CUDA Toolkit with Conda simply requires

```
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

```
conda env create -f environment.yml
conda activate pulseTrain
```

Other useful conda commands

```
conda install PACKAGE_NAME # Install a package
conda deactivate # Exit the environment
conda env list # List all conda environments
conda create --name ENV_NAME # Create a new conda environment carrying over dependencies from base
conda env export > environment.yml # Export active environment to a configuration file
conda remove --name ENV_NAME --all # deletes environment ENV_NAME and uninstalls associated packages
```

## Contributing

For additional documentation and contribution guidelines see [CONTRIBUTING.md](CONTRIBUTING.md)
