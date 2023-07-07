# Contributing

# Server setup

This documents the process for throwing together a server on a Windows 10 PC with a 3070TI

## Git Bash and Conda install

See [README.md](README.md)

## CUDA install

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

## PyTorch install

Following instructions for [local installation](https://pytorch.org/get-started/locally/), run

```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

At time of install pytorch was not available for Cuda 12, and running the conda install downgraded the cuda version automatically

To verify the install, launch a python instance in conda

```python
import torch
torch.rand(5, 3) # Should output a tensor
torch.cuda.is_available() # Check if GPU driver and CUDA is enabled
```

## NoMachine

Follow instructions from the [NoMachine website](https://www.nomachine.com/getting-started-with-nomachine) to enable remote desktop access from another computer

## SSH windows server

NoMachine is fast to set up but annoying as an actual interface for development. Here we describe how to configure OpenSSH via Powershell

1. Enable OpenSSH _Server_ on windows. (Instructions [here](https://learn.microsoft.com/en-us/windows-server/administration/openssh/openssh_install_firstuse?tabs=powershell), but if this does not work opt for [manual installation](https://www.saotn.org/manually-install-openssh-in-windows-server/)). Instructions that follow assume manual installation on powershell was followed.

To run the file `install-sshd.ps1` use

```
powershell -ExecutionPolicy Bypass -File install-sshd.ps1
```

To add the firewall rule use

```
New-NetFirewallRule -Name sshd -DisplayName 'OpenSSH SSH Server' -Enabled True -Direction Inbound -Protocol TCP -Action Allow -LocalPort 22 -Program "C:\OpenSSH-64\sshd.exe"
```

2. Validate that the sshd service is running and waiting on port 22

```
netstat -nao | find /i '":22"'
```

3. Get your local account information to obtain `computer_name/my_username`

```
whoami
```

4. Test ssh access on the local host (i.e., same computer)

```
ssh -l my_username localhost # Powershell
```

5. Get the IP_address for ssh with `ipconfig`. Then, for example, from a linux instance one simply needs to run

```
ssh my_username@IP_address
```

1. Change the ssh terminal from Windows CMD to Git Bash, following the methodology outlined [here](https://learn.microsoft.com/en-us/windows-server/administration/openssh/openssh_server_configuration)

```
New-ItemProperty -Path "HKLM:\SOFTWARE\OpenSSH" -Name DefaultShell -Value "C:\Program Files\Git\bin\bash.exe" -PropertyType String -Force
```

### Uninstalling sshd

In `C:\OpenSSH-Win64`

```
powershell -ExecutionPolicy Bypass -File uninstall-sshd.ps1
```
