#!/bin/bash

# Configurar swap de memória na Jetson +2Gb
echo "Configure swap de memória adicional para Jetson (+2Gb min.)"

# Informar o CUDA Compute Capability, Jetson 5.3, Desktop 7.5
echo "CUDA Compute Capabilitiy:"
echo "Jetson Nano               | 5.3"
echo "Jetson TX1                | 5.3"
echo "Jetson Tegra X1           | 5.3"
echo "Jetson TX2                | 6.2"
echo "Jetson AGX Xavier         | 7.2"
echo "GTX 1660 Super            | 7.5"
echo "GTX 1060                  | 6.1"
echo "Consulte https://www.myzhar.com/blog/tutorials/tutorial-nvidia-gpu-cuda-compute-capability/"
read -rp "Informe CUDA Compute Capability da sua placa: " capability
read -rp "Continuar (s/n)?" pass

if [ $pass != "s" ]; then
    exit 0
fi

# Libera memória ram utilizando interface desktop mais leve
free -m

# Jetson Nano
read -rp "Arquitetura ARM (Jetson Nano) (s/n)?" jetnano
if [ $jetnano == "s" ]; then
    # Variável indicando arquitetura arm, bug opencv
    export 'OPENBLAS_CORETYPE=ARMV8' >> ~/.bashrc

    # Revela localização CUDA
    sudo sh -c "echo '/usr/local/cuda/lib64' >> /etc/ld.so.conf.d/nvidia-tegra.conf"
    sudo ldconfig

fi

# Remove OpenCV anterior
sudo apt purge libopencv-dev libopencv-python libopencv-samples libopencv*

# download the latest version
cd ~
wget -nc -O opencv.zip https://github.com/opencv/opencv/archive/4.5.5.zip
wget -nc -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.5.5.zip
# unpack
unzip opencv.zip
unzip opencv_contrib.zip
# some administration to make live easier later on
mv opencv-4.5.5 opencv
mv opencv_contrib-4.5.5 opencv_contrib

cd ~/opencv
mkdir build
cd build

# Incluir #-D ENABLE_NEON=ON \ para Jetson quando tiver suporte
cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
    -D EIGEN_INCLUDE_PATH=/usr/include/eigen3 \
    -D WITH_OPENCL=OFF \
    -D WITH_CUDA=ON \
    -D CUDA_ARCH_BIN=$capability \
    -D CUDA_ARCH_PTX="" \
    -D WITH_CUDNN=ON \
    -D WITH_CUBLAS=ON \
    -D ENABLE_FAST_MATH=ON \
    -D CUDA_FAST_MATH=ON \
    -D OPENCV_DNN_CUDA=ON \
    -D WITH_QT=OFF \
    -D WITH_OPENMP=ON \
    -D WITH_OPENGL=ON \
    -D BUILD_TIFF=ON \
    -D WITH_FFMPEG=ON \
    -D WITH_GSTREAMER=ON \
    -D WITH_TBB=ON \
    -D BUILD_TBB=ON \
    -D BUILD_TESTS=OFF \
    -D WITH_EIGEN=ON \
    -D WITH_V4L=ON \
    -D WITH_LIBV4L=ON \
    -D OPENCV_ENABLE_NONFREE=ON \
    -D INSTALL_C_EXAMPLES=OFF \
    -D INSTALL_PYTHON_EXAMPLES=OFF \
    -D BUILD_NEW_PYTHON_SUPPORT=ON \
    -D BUILD_opencv_python3=TRUE \
    -D OPENCV_GENERATE_PKGCONFIG=ON \
    -D BUILD_EXAMPLES=OFF \
    -D PYTHON3_EXECUTABLE=$(which python3) \
    -D PYTHON_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
    -D PYTHON_INCLUDE_DIR2=$(python3 -c "from os.path import dirname; from distutils.sysconfig import get_config_h_filename; print(dirname(get_config_h_filename()))") \
    -D PYTHON_LIBRARY=$(python3 -c "from distutils.sysconfig import get_config_var;from os.path import dirname,join ; print(join(dirname(get_config_var('LIBPC')),get_config_var('LDLIBRARY')))") \
    -D PYTHON3_NUMPY_INCLUDE_DIRS=$(python3 -c "import numpy; print(numpy.get_include())") \
    -D PYTHON3_PACKAGES_PATH=$(python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") ..

NPROC=$(nproc)
make -j$NPROC
sudo make install
sudo ldconfig

# cleaning
make clean
cd ~
rm -rf opencv*.zip