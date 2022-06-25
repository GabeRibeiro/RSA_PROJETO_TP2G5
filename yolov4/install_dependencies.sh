#!/bin/bash
#
# Reference for installing 'pycuda': https://wiki.tiker.net/PyCuda/Installation/Linux/Ubuntu

# 0. In case nvcc is not found
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64\
                         ${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

################################################################################
# 1. Install protobuf
#sudo -H pip3 install jproperties protobuf numpy scipy

set -e

folder=${HOME}/src
mkdir -p $folder

echo "** Install requirements"
sudo apt-get install -y autoconf libtool

echo "** Download protobuf-3.8.0 sources"
cd $folder
if [ ! -f protobuf-python-3.8.0.zip ]; then
  wget https://github.com/protocolbuffers/protobuf/releases/download/v3.8.0/protobuf-python-3.8.0.zip
fi
if [ ! -f protoc-3.8.0-linux-aarch_64.zip ]; then
  wget https://github.com/protocolbuffers/protobuf/releases/download/v3.8.0/protoc-3.8.0-linux-aarch_64.zip
fi

echo "** Install protoc"
unzip protobuf-python-3.8.0.zip
unzip protoc-3.8.0-linux-aarch_64.zip -d protoc-3.8.0
sudo cp protoc-3.8.0/bin/protoc /usr/local/bin/protoc

echo "** Build and install protobuf-3.8.0 libraries"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=cpp
cd protobuf-3.8.0/
./autogen.sh
./configure --prefix=/usr/local
make -j$(nproc)
make check
sudo make install
sudo ldconfig

echo "** Update python3 protobuf module"
# remove previous installation of python3 protobuf module
sudo apt-get install -y python3-pip
sudo pip3 uninstall -y protobuf
sudo pip3 install Cython
cd python/
python3 setup.py build --cpp_implementation
python3 setup.py test --cpp_implementation
sudo python3 setup.py install --cpp_implementation

echo "** Build protobuf-3.8.0 successfully"

################################################################################
# 2. Install pycuda
set -e

if ! which nvcc > /dev/null; then
  echo "ERROR: nvcc not found"
  exit
fi

arch=$(uname -m)
folder=${HOME}/src
mkdir -p $folder

echo "** Install requirements"
sudo apt-get install -y build-essential python3-dev
sudo apt-get install -y libboost-python-dev libboost-thread-dev
sudo pip3 install setuptools

boost_pylib=$(basename /usr/lib/${arch}-linux-gnu/libboost_python*-py3?.so)
boost_pylibname=${boost_pylib%.so}
boost_pyname=${boost_pylibname/lib/}

echo "** Download pycuda-2019.1.2 sources"
pushd $folder
if [ ! -f pycuda-2019.1.2.tar.gz ]; then
  wget https://files.pythonhosted.org/packages/5e/3f/5658c38579b41866ba21ee1b5020b8225cec86fe717e4b1c5c972de0a33c/pycuda-2019.1.2.tar.gz
fi

echo "** Build and install pycuda-2019.1.2"
CPU_CORES=$(nproc)
echo "** cpu cores available: " $CPU_CORES
tar xzvf pycuda-2019.1.2.tar.gz
cd pycuda-2019.1.2
python3 ./configure.py --python-exe=/usr/bin/python3 --cuda-root=/usr/local/cuda --cudadrv-lib-dir=/usr/lib/${arch}-linux-gnu --boost-inc-dir=/usr/include --boost-lib-dir=/usr/lib/${arch}-linux-gnu --boost-python-libname=${boost_pyname} --boost-thread-libname=boost_thread --no-use-shipped-boost
make -j$CPU_CORES
python3 setup.py build
sudo python3 setup.py install

popd

python3 -c "import pycuda; print('pycuda version:', pycuda.VERSION)"

################################################################################
# 3. Install onnx

# Fix for the python3.7 error
sudo pip3 install typing-extensions==3.6.2.1
#onnx
sudo pip3 install onnx==1.9.0

