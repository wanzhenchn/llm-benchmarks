#! /usr/bin/bash

mkdir -p build && cd build

SM="80;89;90"
cmake .. \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
    -DCMAKE_INSTALL_PREFIX=./install \
    -DBUILD_PY_FFI=ON \
    -DBUILD_MULTI_GPU=ON \
    -DCMAKE_CUDA_FLAGS="-lineinfo" \
    -DUSE_NVTX=ON \
    -DSM=${SM} -DCMAKE_CUDA_ARCHITECTURES=${SM}

make -j$(nproc) && make install

cd .. &&

pip install .
