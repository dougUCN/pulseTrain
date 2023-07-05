# pulseTrain

## Getting started

Instructions are for a Windows 10 PC with a 3070TI, Git bash and Conda

Clone this repository

```
git clone https://github.com/dougUCN/pulseTrain.git
```

### Git Bash

Install the Git Bash utility for Windows from [Git SCM](https://git-scm.com/downloads)

### Conda

Install Conda following the instructions [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)

To get Conda to run on Git Bash, to the `~/.bashrc` configuration file add the line

```
source /c/Users/$USERNAME/anaconda3/etc/profile.d/conda.sh
```

with `$USERNAME` replaced appropriately, or the path modified if you chose a different installation directory for conda. Then, restart your Git Bash instance.

Create a conda environment from the `environment.yml` file

```
conda env create -f environment.yml
conda activate pulseTrain
```

Other useful conda commands

```
conda create --name ENV_NAME # Create a new conda environment
conda env export > environment.yml # Export active environment to a configuration file
conda install PACKAGE_NAME # Install a package
conda deactivate # Exit the environment
conda env list # List all conda environments
conda remove --name ENV_NAME --all # deletes environment ENV_NAME and uninstalls associated packages
```

## Git LFS

The coincidences file (`coinc_4200_5441.dat`) is large and requires the use of the Git large file storage (LFS). GitHub allows only 1 Gb/mo of bandwidth, which includes cloning of the file.

## Contributing

For additional documentation and contribution guidelines see [CONTRIBUTING.md](CONTRIBUTING.md)
