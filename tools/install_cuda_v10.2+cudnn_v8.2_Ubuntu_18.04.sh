#!/bin/bash

# Press Ctrl + Alt + F3 to enter tty mode and login as usual user.
echo ""
sudo service lightdm stop
echo ""
sudo service gdm3 stop

### If you have previous installation remove it first. 
# Cuda Toolkit
sudo apt-get --purge remove "*cublas*" "*cufft*" "*curand*" \
 "*cusolver*" "*cusparse*" "*npp*" "*nvjpeg*" "cuda*" "nsight*"
# Nvidia drivers
sudo apt-get --purge remove "*nvidia*"
# Xserver config
sudo rm /etc/X11/xorg.conf
# Others
sudo apt-get purge nvidia*
sudo apt remove nvidia-*
sudo rm /etc/apt/sources.list.d/cuda*
sudo apt-get autoremove && sudo apt-get autoclean
sudo rm -rf /usr/local/cuda*
sudo apt-get purge cuda-*
sudo apt-get remove cuda-*
# Regenerate the kernel initramfs
sudo update-initramfs -u

### to verify your gpu is cuda enable check
lspci | grep -i nvidia

### gcc compiler is required for development using the cuda toolkit. to verify the version of gcc install enter
gcc --version

# system update
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install linux-headers-$(uname -r)

# download cuda_toolkit and installing CUDA-10.2
wget -nc https://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_440.33.01_linux.run
sudo sh cuda_10.2.89_440.33.01_linux.run

# setup your paths
echo 'export PATH=/usr/local/cuda-10.2/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
sudo ldconfig

# install cuDNN v8
# in order to download cuDNN you have to be regeistered here https://developer.nvidia.com/developer-program/signup
# then download cuDNN v8 form https://developer.nvidia.com/cudnn
CUDNN_TAR_FILE="cudnn-10.2-linux-x64-v8.2.2.26.tgz"
# Baixar pelo site separado, wget sem resposta, provavelmente por precisar logar
# wget -nc https://developer.nvidia.com/compute/machine-learning/cudnn/secure/8.2.2/10.2_07062021/cudnn-10.2-linux-x64-v8.2.2.26.tgz
tar -xzvf ${CUDNN_TAR_FILE}

# copy the following files into the cuda toolkit directory.
sudo cp -P cuda/include/* /usr/local/cuda-10.2/include
sudo cp -P cuda/lib64/* /usr/local/cuda-10.2/lib64/
sudo chmod a+r /usr/local/cuda-10.2/lib64/*