# Chapter 6 GPU Acceleration

Apart from the significant acceleration capabilities on Intel CPUs, BigDL-LLM also supports optimizations and acceleration for running LLMs (large language models) on Intel GPUs.

BigDL-LLM supports optimizations of any [*HuggingFace transformers*](https://huggingface.co/docs/transformers/index) model on Intel GPUs with the help of low-bit techniques, modern hardware accelerations and latest software optimizations.

#### 6B model running on Intel Arc GPU (real-time screen capture):

<p align="left">
            <img src="https://llm-assets.readthedocs.io/en/latest/_images/chatglm2-arc.gif" width='60%' /> 

</p>

#### 13B model running on Intel Arc GPU (real-time screen capture): 

<p align="left">
            <img src="https://llm-assets.readthedocs.io/en/latest/_images/llama2-13b-arc.gif" width='60%' /> 

</p>

In Chapter 6, you will learn how to run LLMs, as well as implement stream chat functionalities, using BigDL-LLM optimizations on Intel GPUs. Popular open source models are used as examples:

+ [Llama2-7B](./6_1_GPU_Llama2-7B.md)

## 6.0 Environment Setup

**The following sub-sectors give a rough example of how to setup environment on Data Center GPU, Linux system, and PyTorch 2.0. For more and detailed installation instructions, please refer to [environment setup page](environment_setup.md) or the GPU installation [Webpage](https://bigdl.readthedocs.io/en/latest/doc/LLM/Overview/install_gpu.html)**

### 6.0.1 System Recommendation

For a smooth experience with the notebooks in Chapter 7, please ensure your hardware and OS meet the following requirements:

> ⚠️Hardware
  - Intel Arc™ A-Series Graphics
  - Intel Data Center GPU Flex Series
  - Intel Data Center GPU Max Series

> ⚠️Operating System
  - Linux system, Ubuntu 22.04 is preferred

    > **Note**
    > Please note that both Linux and Windows OS are supported for BigDL-LLM optimizations on Intel GPUs.
  


### 6.0.2 Driver and Toolkit Installation

Before benefiting from BigDL-LLM on Intel GPUs, there are several steps for tools installation:

- First you need to install Intel GPU driver. Please refer to our [driver installation](https://dgpu-docs.intel.com/driver/installation.html) for general purpose GPU capabilities.
  > **Note**
  > For BigDL-LLM with default IPEX version (IPEX 2.0.110+xpu), Intel GPU Driver version [Stable 647.21](https://dgpu-docs.intel.com/releases/stable_647_21_20230714.html) is required.

- You also need to download and install [Intel® oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html). OneMKL and DPC++ compiler are needed, others are optional.
  > **Note**
  > BigDL-LLM with default IPEX version (IPEX 2.0.110+xpu) requires Intel® oneAPI Base Toolkit's version == 2023.2.0.

### 6.0.3 Python Environment Setup

Next, use a python environment management tool (we recommend using [Conda](https://docs.conda.io/projects/conda/en/stable/)) to create a python environment and install necessary libs.

#### 6.0.3.1 Install Conda

For Linux users, open a terminal and run below commands:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ./Miniconda3-latest-Linux-x86_64.sh
conda init
```

> **Note**
> Follow the instructions popped up on the console until conda initialization finished successfully.

#### 6.0.3.2 Create Environment

> **Note**
> Python 3.9 is recommended for running BigDL-LLM.

Create a Python 3.9 environment with the name you choose, for example `llm-tutorial-gpu` and install `bigdl-llm` from PyPI:

```bash
conda create -n llm-tutorial-gpu python=3.9
conda activate llm-tutorial-gpu   # install ipex 2.0 and pytorch 2.0 in this environment

pip install --pre --upgrade bigdl-llm[xpu] -f https://developer.intel.com/ipex-whl-stable-xpu
```

### 6.0.4 Best Known Configuration on Linux

For optimal performance on Intel GPUs, it is recommended to set several environment variables:

```bash
# configure oneAPI environment variables
source /opt/intel/oneapi/setvars.sh

export USE_XETLA=OFF
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
```
