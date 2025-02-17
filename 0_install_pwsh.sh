#!/usr/bin/bash
#export PIP_INDEX_URL="https://pypi.mirrors.ustc.edu.cn/simple"
#export HF_ENDPOINT="https://hf-mirror.com"

echo "Checking if PowerShell is installed..."
if ! command -v pwsh &> /dev/null
then
    echo "PowerShell is not installed, installing now..."
    
    # Update package list
    sudo apt-get update
    
    # Install prerequisites
    sudo apt-get install -y wget apt-transport-https software-properties-common
    
    # Get Ubuntu version
    source /etc/os-release
    
    # Download and register Microsoft repository keys
    wget -q https://packages.microsoft.com/config/ubuntu/$VERSION_ID/packages-microsoft-prod.deb
    sudo dpkg -i packages-microsoft-prod.deb
    rm packages-microsoft-prod.deb
    
    # Update package list with new repository
    sudo apt-get update
    
    # Install PowerShell
    sudo apt-get install -y powershell
    
    echo "PowerShell installation completed"
else
    echo "PowerShell is already installed"
fi

echo "Install completed"
