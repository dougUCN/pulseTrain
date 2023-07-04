# Contributing

# Server setup

This documents the process for throwing together a server on a Windows 10 PC with a 3070TI

## Git Bash

Install the Git Bash utility from [Git SCM](https://git-scm.com/downloads)

## NoMachine

Follow instructions from the [NoMachine website](https://www.nomachine.com/getting-started-with-nomachine) to enable remote desktop access from another computer

## SSH windows server

NoMachine is fast to set up but annoying as an actual interface for development. Here we describe how to e

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

6. Change the ssh terminal from Windows CMD to Git Bash. Following the methodology outlined [here](https://learn.microsoft.com/en-us/windows-server/administration/openssh/openssh_server_configuration)

```
New-ItemProperty -Path "HKLM:\SOFTWARE\OpenSSH" -Name DefaultShell -Value "C:\Program Files\Git\bin\bash.exe" -PropertyType String -Force
```

### Uninstalling sshd

In `C:\OpenSSH-Win64`

```
powershell -ExecutionPolicy Bypass -File uninstall-sshd.ps1
```

## Conda environment
