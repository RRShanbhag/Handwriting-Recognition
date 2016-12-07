#!/usr/bin/env bash

git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git
cd opencv
mkdir build
cd build

cmake -D CMAKE_BUILD_TYPE=RELEASE \
        -D CMAKE_INSTALL_PREFIX=$HOME/Applications/opencv \
        -D INSTALL_C_EXAMPLES=OFF \
        -D INSTALL_PYTHON_EXAMPLES=ON \
        -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
        -D PYTHON3_EXECUTABLE=/home/varunbhat/workspace/envs/ml_prj/bin/python \
        -D BUILD_EXAMPLES=ON ..


make -j8
make install

source $HOME/workspace/envs/ml_prj/bin/activate
cd $HOME/workspace/ml_project
export LD_LIBRARY_PATH=$HOME/Applications/opencv/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$PYTHONPATH:$HOME/Applications/opencv/lib/python3.5/site-packages


