# 基于Intel Xeon和SPR的大语言模型应用开发指南

本文档介绍如何开发大语言模型应用UI，基于开源的intel bigdl-llm库和gradio。
UI跑在windows11 x86 CPU上或者Ubuntu CPU上，实现在6根16GB内存以上运行优化的Native INT4 大语言模型。
以三个大语言模型为例，ChatGLM2 (6B)中英，LLaMA2 (13B)英。

Note: 通常客户服务器出货系统，CPU的内存槽位不会插满，这种情况下必须要根据实际条数插到特定的槽位，否则会影响到系统稳定性与性能。以下是Eagle Stream内存插法示意图。
![image](https://github.com/KiwiHana/bigdl-llm-tutorial/assets/102839943/a54c74cc-6581-4f9e-b2b4-3780bbcfe2a6)


## 1 安装环境
（1）Windows11安装Miniconda3-py39_23.5.2-0-Windows-x86_64.exe，下载链接：
https://docs.conda.io/en/latest/miniconda.html#windows-installers 

如果是Ubuntu,下载安装
```
wget https://repo.anaconda.com/miniconda/Miniconda3-py39_23.5.2-0-Linux-x86_64.sh
chmod -R 777 Miniconda3-py39_23.5.2-0-Linux-x86_64.shy
./Miniconda3-py39_23.5.2-0-Linux-x86_64.sh
sudo apt install numactl
```

（2）打开Anaconda Powershell Prompt窗口
```
conda create -n llm python=3.9
conda activate llm
pip install --pre --upgrade bigdl-llm[all]
pip install bigdl-nano
pip install gradio==3.41.1 mdtex2html
```
或者用指定版本的方式安装
```
 pip install --pre bigdl-llm[all]==2.4.0b20231110 -i https://pypi.tuna.tsinghua.edu.cn/simple
```
## 2 LLM模型转换
以Chatglm2，llama2，starcoder为例，下载hugging face FP16模型。模型下载链接：

· ChatGLM2-6B：https://huggingface.co/THUDM/chatglm2-6b/tree/main

· Llama2-13B: https://huggingface.co/meta-llama/Llama-2-13b-chat-hf/tree/main


### 2.1 FP16转Native INT4模型，并调用python函数 (推荐运行在CPU)
Chatglm2 ，llama2 转native INT4。

打开Anaconda PowerShell，修改模型路径和输出文件夹名称，并运行：
```
 conda activate llm
 llm-convert "/llm-models/chatglm2-6b/" --model-format pth --model-family "chatglm" --outfile "checkpoint/"
 llm-convert "/llm-models/llama-2-13b-chat-hf/" --model-format pth --model-family "llama" --outfile "checkpoint/"
```


#### python调用Native INT4模型。
参数解释：

（1）n_threads
对于Xeon，OMP_NUM_THREADS 和 n_threads是第一个socket的物理核数量。
对于有2个socket的SPR，需要指定用第一个socket所有物理核进行推理。

（2）n_ctx=4096表示模型最长的输入+输出文本等于4096 tokens
```
from bigdl.llm.ggml.model.chatglm.chatglm import ChatGLM
from bigdl.llm.transformers import BigdlNativeForCausalLM
model_name = "chatglm2-6b"
model_all_local_path = "C:\\PC_LLM\\checkpoint\\"
if model_name == "chatglm2-6b":
    model = ChatGLM(model_all_local_path + "\\ggml-chatglm2-6b-q4_0.bin", n_threads=20,n_ctx=4096) 

elif model_name == "llama2-13b":
    model = BigdlNativeForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_all_local_path + "\\bigdl_llm_llama2_13b_q4_0.bin",
    model_family='llama',n_threads=20,n_ctx=4096)
```
### 2.2 FP16转transformer INT4，并调用python函数
Transformer INT4在CPU上运行性能比Native INT4低一些。

用python脚本转换模型为transformer INT4
```
from bigdl.llm.transformers import AutoModel
from transformers import AutoTokenizer
from bigdl.llm.transformers import AutoModelForCausalLM
model_name = "chatglm2-6b"
model_all_local_path = "C:\\PC_LLM\\checkpoint\\"
model_name_local = model_all_local_path + model_name

if model_name == "chatglm2-6b":
    tokenizer = AutoTokenizer.from_pretrained(model_name_local, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name_local, trust_remote_code=True, load_in_4bit=True)
    model.save_low_bit("D:\\llm-models\\chatglm2-6b-int4\\")
    tokenizer.save_pretrained("D:\\llm-models\\chatglm2-6b-int4\\")

elif model_name == "llama2-13b":
    tokenizer = AutoTokenizer.from_pretrained(model_name_local, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name_local, trust_remote_code=True, load_in_4bit=True)
    model.save_low_bit("D:\\llm-models\\"+model_name)
    tokenizer.save_pretrained("D:\\llm-models\\"+model_name)
```
python调用transformer INT4模型
```
if model_name == "chatglm2-6b":
    model = AutoModel.load_low_bit(model_name_local,trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name_local,trust_remote_code=True)
    model = model.eval()
elif model_name == "llama2-13b":
    model = AutoModelForCausalLM.load_low_bit(model_name_local,trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name_local,trust_remote_code=True)
    model = model.eval()
```
## 3 测试LLM benchmark on CPU
使用Native INT4模型测试LLM benchmark on CPU将会使用所有核，方便和应用UI的性能指标相比较。

对于Ubuntu平台，假设SPR第一个socket有48个物理核。numactl -C 0-47 -m 0 $command
```
$ lscpu
NUMA node0 CPU(s): 0-47,96-143
NUMA node1 CPU(s): 48-95,144-191

Therefore, you will set parameters like:
$ export OMP_NUM_THREADS=48
$ numactl -C 0-47 -m 0 llm-cli -t 48 ……
```

```
sudo apt install numactl
conda create -n llm python=3.9
conda activate llm
pip install bigdl-llm[all]
pip install bigdl-nano
source bigdl-nano-init -c
export OMP_NUM_THREADS=48
export TRANSFORMERS_OFFLINE=1
$ numactl -C 0-47 -m 0 llm-cli -t 48 -x chatglm -m "./checkpoint/bigdl_llm_chatglm_q4_0.bin" -p "Once upon
a time, there existed a little girl who liked to have adventures. She wanted to go to places and meet new
people, and have fun" --no-mmap -v -n 32

numactl -C 0-47 -m 0 llm-cli -t 48 -x llama -m "bigdl_llm_llama2_13b_q4_0.bin" -p "Once upon a time, there existed a little girl who liked to have adventures. She wanted to go to places and meet new people, and have fun" --no-mmap -n 32
```

对于windows平台，假设SPR第一个socket有48个物理核。start /node 0 $command
```
conda create -n llm python=3.9
conda activate llm
pip install bigdl-llm[all]
pip install bigdl-nano

start /node 0 llm-cli -t 48 -x chatglm -m "./checkpoint/bigdl_llm_chatglm_q4_0.bin" -p "Once upon
a time, there existed a little girl who liked to have adventures. She wanted to go to places and meet new
people, and have fun" --no-mmap -v -n 32
```

参数解释：-n 32限制输出32 tokens。

从command line提取性能信息，如图：

Input token: 32 tokens 

Output token: 32 tokens (31 runs = 32 tokens – 1st token)

1st token avg latency (ms) = 1541.56 ms 

2nd+ token avg latency (ms/token) = 125.62 ms per token

图1：llm-cli的输出
![image](https://github.com/KiwiHana/bigdl-llm-tutorial/assets/102839943/5adf144a-5fc5-432f-b476-f26d341fbced)


## 4 用吐字的方式输出文本
### 4.1（推荐运行在CPU）：Native int4 for chatglm2，llama2和starcoder。
```
from bigdl.llm.ggml.model.chatglm.chatglm import ChatGLM
model_name = "chatglm2-6b"
model_all_local_path = "C:\\PC_LLM\\checkpoint\\"
prompt = "What is AI?"
if model_name == "chatglm2-6b":
    model = ChatGLM(model_all_local_path + "\\ggml-chatglm2-6b-q4_0.bin", n_threads=20,n_ctx=4096)
    response = ""
    for chunk in model(prompt, temperature=0.95,top_p=0.8,stream=True,max_tokens=512):
        response += chunk['choices'][0]['text']
```
llama2和starcoder的吐字调用方式相同，也是用for循环。

参数说明：

· 温度（Temperature）（数值越高，输出的随机性增加），可调范围0~1

· Top P（数值越高，词语选择的多样性增加），可调范围0~1

· 输出最大长度（Max Length）（输出文本的最大tokens），可调范围0~2048，上限由模型决定。这三个模型n_ctx最大8k，输入+输出tokens应小于8k。

### 4.2 Transformer INT4 stream_chat仅限chatglm2
```
from bigdl.llm.transformers import AutoModel
from transformers import AutoTokenizer
import torch
model_name = "chatglm2-6b"
model_all_local_path = "C:\\PC_LLM\\checkpoint\\"
model_name_local = model_all_local_path + model_name
prompt = "What is AI?"
model = AutoModel.load_low_bit(model_name_local,trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name_local,trust_remote_code=True)
model = model.eval()
with torch.inference_mode():
    for response, history in model.stream_chat(tokenizer, prompt, history,max_length=512, top_p=0.9,temperature=0.9):
        print(response)
```
### 4.3 Transformer INT4 TextIteratorStreamer for chatglm2，llama2和starcoder。
```
from bigdl.llm.transformers import AutoModel
from transformers import AutoTokenizer,TextIteratorStreamer
import torch
from benchmark_util import BenchmarkWrapper

model_name = "chatglm2-6b"
model_all_local_path = "C:\\PC_LLM\\checkpoint\\"
model_name_local = model_all_local_path + model_name
model = AutoModel.load_low_bit(model_name_local,trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name_local,trust_remote_code=True)
model = model.eval()
prompt = "What is AI?"
with torch.inference_mode():
    model=BenchmarkWrapper(model)
    inputs = tokenizer(prompt, return_tensors="pt")
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    response = ""
    timeStart = time.time() 
  #  out = model.generate(**inputs, streamer=streamer, temperature=0.9, top_p=0.9, max_new_tokens=512) 
    generate_kwargs = dict(**inputs,streamer=streamer,temperature=0.9, top_p=0.9, max_new_tokens=512)
    from threading import Thread
    thread = Thread(target=model.generate, kwargs=generate_kwargs)
    thread.start()

    for new_text in streamer:
        response += new_text
    timeCost = time.time() - timeStart
    token_count_input = len(tokenizer.tokenize(prompt))

```

## 5 添加history多轮对话功能
### 5.1 仅对chatglm2 Transformer INT4 stream_chat
代码参考4.2
### 5.2对于Native int4添加history多轮对话功能
```
from bigdl.llm.ggml.model.chatglm.chatglm import ChatGLM
model_name = "chatglm2-6b"
model_all_local_path = "C:\\PC_LLM\\checkpoint\\"
history_round = 0
history = []
if model_name == "chatglm2-6b":
    model = ChatGLM(model_all_local_path + "\\ggml-chatglm2-6b-q4_0.bin", n_threads=20,n_ctx=4096)
input = "你好"
predict(input)
input = "请进行丽江三天必游景点旅游规划"
predict(input)

def predict(input):
    global history_round, model, history
    response = ""
    if len(model.tokenize(history)) > 2500 or history_round >= 5: ### history record 5 rounds
        history_round = 0
        history = [] 
        print("*********** reset chatbot and history", history)

    if len(history) == 0:
        print("*********** new chat ")
        prompt = input
        history = prompt
        history_round = 1
    else:
        prompt = history + '\n' + input
        history_round += 1
    print("******************* history_round ", history_round)

    timeStart = time.time()
    for chunk in model(prompt, temperature=0.9,top_p=0.9, stream=True,max_tokens=512):
        response += chunk['choices'][0]['text']
    history = prompt + response
    print("******** max_length history",len(model.tokenize(history)))
```
### 5.3 对于transformer INT4 TextIteratorStreamer同5.2 
## 6 用gradio写Web UI
下载代码：https://github.com/KiwiHana/LLM_UI_Windows_CPU

![image](https://github.com/KiwiHana/bigdl-llm-tutorial/assets/102839943/5a399c7e-31b4-4337-a6a4-bc6f8bccb93c)
图2：LLM_UI_Windows_CPU界面


为了使用全部核，用管理员打开Anaconda Powershell Prompt窗口，运行LLM_demo_v1.0.py 或 LLM_demo_v2.0.py。
```
git clone https://github.com/KiwiHana/LLM_UI_Windows_CPU.git
cd LLM_UI_Windows_CPU
conda activate llm
python LLM_demo_v1.0.py
```
Note: 修改LLM_demo_v1.0.py脚本第285行 main函数里的模型存放路径，

例如 model_all_local_path = "C:/Users/username/checkpoint/"

· 大语言模型应用UIv1.0文件夹应包含：

LLM_demo_v1.0.py

theme3.json

checkpoint

-- bigdl_llm_llama2_13b_q4_0.bin

-- ggml-chatglm2-6b-q4_0.bin


参考链接：

https://github.com/intel-analytics/bigdl-llm-tutorial/tree/main/ch_2_Environment_Setup
