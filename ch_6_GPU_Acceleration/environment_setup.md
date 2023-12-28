## Environment setup for Intel Arc GPU

## Linux

**Prerequisites**: BigDL-LLM on Linux supports Intel Arc™ A-Series Graphics, Intel Data Center GPU Flex Series, and Intel Data Center GPU Max Series. In addition, we recommend using Python 3.9 since `bigdl-llm` has been tested with this version.

**Importance:** We currently support the Ubuntu 20.04 operating system and later.


### 1. Downgrade kernels
For Linux users, Ubuntu 22.04 and Linux kernel 5.19.0 are preferred. Ubuntu 22.04 and Linux kernel 5.19.0-41-generic are mostly used in our test environment. But the default Linux kernel of Ubuntu 22.04.3 is 6.2.0-35-generic, so we recommend you downgrade the kernel to 5.19.0-41-generic to achieve the best performance.

Here are the steps to downgrade your kernel:
```bash
# downgrade kernel to 5.19.0-41-generic
  
sudo apt-get update && sudo apt-get install  -y --install-suggests  linux-image-5.19.0-41-generic

sudo sed -i "s/GRUB_DEFAULT=.*/GRUB_DEFAULT=\"1> $(echo $(($(awk -F\' '/menuentry / {print $2}' /boot/grub/grub.cfg \
| grep -no '5.19.0-41' | sed 's/:/\n/g' | head -n 1)-2)))\"/" /etc/default/grub

sudo  update-grub

sudo reboot
# As 5.19's kernel doesn't has any arc graphic driver. The machine may not start the desktop correctly, but we can use the ssh to login. 
# Or you can select 5.19's recovery mode in the grub, then choose resume to resume the normal boot directly.
```
**Notice:  As 5.19's kernel doesn't have right Arc graphic driver. The machine may not start the desktop correctly, but you can use the ssh to login. Or you can select 5.19's recovery mode in the grub, then choose resume to resume the normal boot directly.**  
You can remove the 6.2.0 kernel if you don't need it. It's an optional step.
```bash 
# remove latest kernel (optional)
sudo apt purge linux-image-6.2.0-*
sudo apt autoremove
sudo reboot
```

### 2. Install GPU driver
For both **PyTorch 2.0** and **PyTorch 2.1**, install Intel GPU Driver version >= stable_775_20_20231219. We highly recommend installing the latest version of intel-i915-dkms using apt. 

If you want to check more details of the driver installation for Data Center or Client GPU on different Linux operating systems, please refer to [this page](https://dgpu-docs.intel.com/driver/installation.html).

**Importance:** driver releases for Intel client GPUs are only validated with the latest Ubuntu Desktop* LTS release (currently Ubuntu 22.04). Here is an example of installing on Ubuntu:

```bash
# install drivers
# setup driver's apt repository
sudo apt-get install -y gpg-agent wget
wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | \
  sudo gpg --dearmor --output /usr/share/keyrings/intel-graphics.gpg
echo "deb [arch=amd64,i386 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu jammy client" | \
  sudo tee /etc/apt/sources.list.d/intel-gpu-jammy.list

sudo apt-get update

sudo apt-get -y install \
    gawk \
    dkms \
    linux-headers-$(uname -r) \
    libc6-dev
	
sudo apt install intel-i915-dkms=1.23.5.19.230406.21.5.17.0.1034+i38-1 intel-platform-vsec-dkms=2023.20.0-21 intel-platform-cse-dkms=2023.11.1-36 intel-fw-gpu=2023.39.2-255~22.04

sudo apt-get install -y gawk libc6-dev udev\
  intel-opencl-icd intel-level-zero-gpu level-zero \
  intel-media-va-driver-non-free libmfx1 libmfxgen1 libvpl2 \
  libegl-mesa0 libegl1-mesa libegl1-mesa-dev libgbm1 libgl1-mesa-dev libgl1-mesa-dri \
  libglapi-mesa libgles2-mesa-dev libglx-mesa0 libigdgmm12 libxatracker2 mesa-va-drivers \
  mesa-vdpau-drivers mesa-vulkan-drivers va-driver-all vainfo
  
sudo reboot

# Configuring permissions

sudo gpasswd -a ${USER} render
newgrp render

# Verify the device is working with i915 driver
sudo apt-get install -y hwinfo
hwinfo --display
```

### 3. Install Intel® oneAPI Base Toolkit
Download and install Intel® oneAPI Base Toolkit from [this page](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html). You can choose different ways of installing like online or offline. Before you install oneAPI, please make sure which PyTorch version you are using. **PyTorch 2.0** and **PyTorch 2.1** require different oneAPI versions. Here is an example installing using APT package manager on Linux for both PyTorch versions: 


**Config oneAPI repository and upgrade APT**
```
wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null
echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list
sudo apt update
```

**PyTorch 2.1** requires oneAPI=2024.0:
```bash
sudo apt install -y intel-basekit # for torch 2.1 and ipex 2.1
```

**PyTorch 2.0** requires oneAPI=2023.2:
```
sudo apt install -y intel-oneapi-common-vars=2023.2.0-49462 \
    intel-oneapi-compiler-cpp-eclipse-cfg=2023.2.0-49495 intel-oneapi-compiler-dpcpp-eclipse-cfg=2023.2.0-49495 \
    intel-oneapi-diagnostics-utility=2022.4.0-49091 \
    intel-oneapi-compiler-dpcpp-cpp=2023.2.0-49495 \
    intel-oneapi-mkl=2023.2.0-49495 intel-oneapi-mkl-devel=2023.2.0-49495 \
    intel-oneapi-mpi=2021.10.0-49371 intel-oneapi-mpi-devel=2021.10.0-49371 \
    intel-oneapi-tbb=2021.10.0-49541 intel-oneapi-tbb-devel=2021.10.0-49541\
    intel-oneapi-ccl=2021.10.0-49084 intel-oneapi-ccl-devel=2021.10.0-49084\
    intel-oneapi-dnnl-devel=2023.2.0-49516 intel-oneapi-dnnl=2023.2.0-49516
```

See the [GPU installation guide](https://bigdl.readthedocs.io/en/latest/doc/LLM/Overview/install_gpu.html) for mode details.

### 4. Install BigDL-LLM and IPEX (Intel Extension for Pytorch)
IPEX 2.1 requries **PyTorch 2.1**, IPEX 2.0 requries **PyTorch 2.0**. We recommend using [miniconda](https://docs.conda.io/projects/miniconda/en/latest/) to create a python 3.9 environment:

#### Install BigDL-LLM From PyPI

**PyTorch 2.1:**
```bash
conda create -n llm python=3.9
conda activate llm

# the [xpu_2.1] indicates we are installing ipex2.1
pip install --pre --upgrade bigdl-llm[xpu_2.1] -f https://developer.intel.com/ipex-whl-stable-xpu
```

**PyTorch 2.0:**
```bash 
conda create -n llm python=3.9
conda activate llm

# the [xpu] indicates we are installing ipex2.0
pip install --pre --upgrade bigdl-llm[xpu] -f https://developer.intel.com/ipex-whl-stable-xpu
```

#### Install BigDL-LLM From Wheel

If you encounter network issues when installing IPEX, you can also install BigDL-LLM dependencies for Intel XPU from source achieves. You need to download and install torch/torchvision/ipex from wheels listed below before installing 'bigdl-llm'.



**PyTorch 2.1:**
First download wheels on your system:
```bash
# get the wheels on Linux system for IPEX 2.1.10+xpu
wget https://intel-extension-for-pytorch.s3.amazonaws.com/ipex_stable/xpu/torch-2.1.0a0%2Bcxx11.abi-cp39-cp39-linux_x86_64.whl
wget https://intel-extension-for-pytorch.s3.amazonaws.com/ipex_stable/xpu/torchvision-0.16.0a0%2Bcxx11.abi-cp39-cp39-linux_x86_64.whl
wget https://intel-extension-for-pytorch.s3.amazonaws.com/ipex_stable/xpu/intel_extension_for_pytorch-2.1.10%2Bxpu-cp39-cp39-linux_x86_64.whl
```

Then, install dependencies directly from the archived wheels and install `bigdl-llm`:
```bash
# install the packages from the wheels
pip install torch-2.1.0a0+cxx11.abi-cp39-cp39-linux_x86_64.whl
pip install torchvision-0.16.0a0+cxx11.abi-cp39-cp39-linux_x86_64.whl
pip install intel_extension_for_pytorch-2.1.10+xpu-cp39-cp39-linux_x86_64.whl

# install bigdl-llm for Intel GPU
# the [xpu_2.1] indicates we are installing ipex2.1
pip install --pre --upgrade bigdl-llm[xpu_2.1]
```

**PyTorch 2.0:**
First download wheels on your system:
```bash
# get the wheels on Linux system for IPEX 2.0.110+xpu
wget https://intel-extension-for-pytorch.s3.amazonaws.com/ipex_stable/xpu/torch-2.0.1a0%2Bcxx11.abi-cp39-cp39-linux_x86_64.whl
wget https://intel-extension-for-pytorch.s3.amazonaws.com/ipex_stable/xpu/torchvision-0.15.2a0%2Bcxx11.abi-cp39-cp39-linux_x86_64.whl
wget https://intel-extension-for-pytorch.s3.amazonaws.com/ipex_stable/xpu/intel_extension_for_pytorch-2.0.110%2Bxpu-cp39-cp39-linux_x86_64.whl
```

Then, install dependencies directly from the archived wheels and install `bigdl-llm`:
```bash
# install the packages from the wheels
pip install torch-2.0.1a0+cxx11.abi-cp39-cp39-linux_x86_64.whl
pip install torchvision-0.15.2a0+cxx11.abi-cp39-cp39-linux_x86_64.whl
pip install intel_extension_for_pytorch-2.0.110+xpu-cp39-cp39-linux_x86_64.whl

# install bigdl-llm for Intel GPU
# the [xpu] indicates we are installing ipex2.0
pip install --pre --upgrade bigdl-llm[xpu]
```

### 5. Runtime Configuration
To use GPU acceleration on Linux, several environment variables are required or recommended before running a GPU example.

For Intel Arc™ A-Series Graphics and Intel Data Center GPU Flex Series, we recommend:
```bash
# configures OneAPI environment variables
source /opt/intel/oneapi/setvars.sh

export USE_XETLA=OFF
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
```

For Intel Data Center GPU Max Series, we recommend:
```bash
# configures OneAPI environment variables
source /opt/intel/oneapi/setvars.sh

export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libtcmalloc.so
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
export ENABLE_SDP_FUSION=1
```
Please note that `libtcmalloc.so` can be installed by `conda install -c conda-forge -y gperftools=2.10`


## Windows

**Prerequisites**: BigDL-LLM on Windows supports Intel iGPU and dGPU.

**Importance**: Currently Windows only supports IPEX(Intel Extension for Pytorch) 2.1, both the following GPU driver and oneAPI installation is only compatible with **PyTorch 2.1**. In addition, we recommend using Python 3.9 since `bigdl-llm` has been tested with this version.

### 1. Install Visual Studio 2022
Install [Visual Studio 2022 Community Edition](https://visualstudio.microsoft.com/downloads/) and select “Desktop development with C++” workload


### 2. Install GPU driver
Install or update to the latest driver, you can download it from [this page](https://www.intel.com/content/www/us/en/download/785597/intel-arc-iris-xe-graphics-windows.html).

### 3. Install Intel® oneAPI Base Toolkit 2024.0

Install from [this page](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html). Choose the Windows operating system and your favored distribution, then you can download the installer(Estimated 2.7GB for the offline one and 17MB for the online one).


### 4. Install BigDL-LLM and IPEX (Intel Extension for Pytorch)

We recommend using [miniconda](https://docs.conda.io/projects/miniconda/en/latest/) to create a python 3.9 environment:

#### Install BigDL-LLM From PyPI

```bash
conda create -n llm python=3.9 libuv
conda activate llm

# the [xpu_2.1] indicates we are installing ipex2.1
pip install --pre --upgrade bigdl-llm[xpu_2.1] -f https://developer.intel.com/ipex-whl-stable-xpu

```

#### Install BigDL-LLM From Wheel

If you encounter network issues when installing IPEX, you can also install BigDL-LLM dependencies for Intel XPU from source achieves. You need to download and install torch/torchvision/ipex from wheels listed below before installing 'bigdl-llm'.

First download wheels on your system:
```bash
wget https://intel-extension-for-pytorch.s3.amazonaws.com/ipex_stable/xpu/torch-2.1.0a0%2Bcxx11.abi-cp39-cp39-win_amd64.whl
wget https://intel-extension-for-pytorch.s3.amazonaws.com/ipex_stable/xpu/torchvision-0.16.0a0%2Bcxx11.abi-cp39-cp39-win_amd64.whl
# here we download ipex2.1
wget https://intel-extension-for-pytorch.s3.amazonaws.com/ipex_stable/xpu/intel_extension_for_pytorch-2.1.10%2Bxpu-cp39-cp39-win_amd64.whl
```

Then, install dependencies directly from the archived wheels and install `bigdl-llm`:
```bash
pip install torch-2.1.0a0+cxx11.abi-cp39-cp39-win_amd64.whl
pip install torchvision-0.16.0a0+cxx11.abi-cp39-cp39-win_amd64.whl
pip install intel_extension_for_pytorch-2.1.10+xpu-cp39-cp39-win_amd64.whl   # here we install ipex2.1

pip install --pre --upgrade bigdl-llm[xpu_2.1]
```

### 5. Runtime Configuration

To use GPU acceleration on Windows, several environment variables are required or recommended before running a GPU example.

Make sure you are using **CMD** as PowerShell is not supported:
```bash
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"

set SYCL_CACHE_PERSISTENT=1

set BIGDL_LLM_XMX_DISABLED=1
```

**Notice: For the first time that each model runs on a new machine, it may take around several minutes to compile.**  

## Helpful Information

Check the [GPU installation guide](https://bigdl.readthedocs.io/en/latest/doc/LLM/Overview/install_gpu.html) for mode details like Known issues.