# 6.1 在英特尔 GPU 上运行 Llama 2 (7B)

您可以使用 BigDL-LLM 加载任何 Hugging Face *transformers* 模型，以便在英特尔 GPU 上加速。有了 BigDL-LLM，Hugging Face 上托管的 PyTorch 模型（FP16/BF16/FP32）可以在英特尔 GPU 上以低位量化（支持的精度包括 INT4/NF4/INT5/INT8）的方式自动加载和优化。

在本教程中，您将学习如何在英特尔 GPU 上运行经过 BigDL-LLM 优化的 LLM，并在此基础上构建一个流式对话机器人。本教程以一个流行的开源 LLM [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)为例。

## 6.1.1 在英特尔 GPU 上安装 BigDL-LLM

首先，在准备好的环境中安装 BigDL-LLM。有关英特尔 GPU 环境设置的最佳做法，请参阅本章的 [README](./README.md#70-environment-setup)。

在终端中运行：

```bash
pip install bigdl-llm[xpu] -f https://developer.intel.com/ipex-whl-stable-xpu
```

> **注意**
> 如果您没有使用上面的安装方式，而是使用了旧版本的`bigdl-llm`(早于`2.5.0b20240104`版本)，您需要在代码开头手动导入`import intel_extension_for_pytorch as ipex`。

完成安装后，您需要为英特尔 GPU 配置 oneAPI 环境变量。

```bash
# 配置 oneAPI 环境变量
source /opt/intel/oneapi/setvars.sh
```

## 6.1.2 (可选) 下载 Llama 2 (7B)

要从 Hugging Face 下载 [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) 模型，您需要获得 Meta 授予的访问权限。请按照 [此处](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf/tree/main) 提供的说明申请模型的访问权限。

获得访问权限后，用您的 Hugging Face token 下载模型：

```python
from huggingface_hub import snapshot_download

model_path = snapshot_download(repo_id='meta-llama/Llama-2-7b-chat-hf',
                               token='hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX') # 将此处改为您自己的 Hugging Face access token
```

> **注意**
> 模型将会默认被下载到 `HF_HOME='~/.cache/huggingface'`.

## 6.1.3 以低精度加载模型

一个常见的用例是以低精度加载 Hugging Face *transformers* 模型，即在加载时进行**隐式**量化。

对于 Llama 2 (7B)，您可以简单地导入 `bigdl.llm.transformers.AutoModelForCausalLM` 而不是 `transformers.AutoModelForCausalLM`，并在 `from_pretrained` 函数中相应地指定 `load_in_4bit=True` 或 `load_in_low_bit` 参数。

对于英特尔 GPU，您应在 `from_pretrained` 函数中**特别设置 `optimize_model=False`** 。**一旦获得低精度模型，请将其设置为 `to('xpu')`**。

**用于 INT4 优化（通过使用 `load_in_4bit=True`）：**

```python
from bigdl.llm.transformers import AutoModelForCausalLM

model_in_4bit = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path="meta-llama/Llama-2-7b-chat-hf",
                                                     load_in_4bit=True,
                                                     optimize_model=False)
model_in_4bit_gpu = model_in_4bit.to('xpu')
```

> **注意**
> BigDL-LLM 支持 `AutoModel`, `AutoModelForCausalLM`, `AutoModelForSpeechSeq2Seq` 以及 `AutoModelForSeq2SeqLM`.
>
> 如果您已经下载了 Llama 2 (7B) 模型并跳过了步骤 [7.1.2.2](#712-optional-download-llama-2-7b)，您可以将`pretrained_model_name_or_path`设置为模型路径。

**(可选) 用于 INT8 优化（通过使用 `load_in_low_bit="sym_int8"`）:**

```python
# 请注意，这里的 AutoModelForCausalLM 是从 bigdl.llm.transformers 导入的
model_in_8bit = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path="meta-llama/Llama-2-7b-chat-hf",
    load_in_low_bit="sym_int8",
    optimize_model=False
)
model_in_8bit_gpu = model_in_8bit.to('xpu')
```

> **注意**
> * 目前英特尔 GPU 上的 BigDL-LLM 支持 `'sym_int4'`, `'asym_int4'`, `'sym_int5'`, `'asym_int5'` 或 `'sym_int8'`选项，其中 'sym' 和 'asym' 用于区分对称量化与非对称量化。选项 `'nf4'` ，也就是 4-bit NormalFloat，同样也是支持的。
>
> * `load_in_4bit=True` 等价于 `load_in_low_bit='sym_int4'`.

## 6.1.4 加载 Tokenizer 

LLM 推理也需要一个 tokenizer. 您可以使用 [Huggingface transformers](https://huggingface.co/docs/transformers/index) API 来直接加载 tokenizer. 它可以与 BigDL-LLM 加载的模型无缝配合使用。对于 Llama 2，对应的 tokenizer 类为 `LlamaTokenizer`.

```python
from transformers import LlamaTokenizer

tokenizer = LlamaTokenizer.from_pretrained(pretrained_model_name_or_path="meta-llama/Llama-2-7b-chat-hf")
```

> **注意**
> 如果您已经下载了 Llama 2 (7B) 模型并跳过了步骤 [7.1.2.2](#712-optional-download-llama-2-7b)，您可以将`pretrained_model_name_or_path`设置为模型路径。

## 6.1.5 运行模型

您可以用与官方 `transformers` API 几乎相同的方式在英特尔 GPU 上使用 BigDL-LLM 优化进行模型推理。**唯一的区别是为 token id 设置 `to('xpu')`**。这里我们为模型创建了一个问答对话模板让其补全。

```python
import torch

with torch.inference_mode():
    prompt = 'Q: What is CPU?\nA:'
    
    # 将输入的 prompt 从字符串转为 token id
    # 使用 .to('xpu') 以在英特尔 GPU 上进行推理
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to('xpu')

    # 基于输入的 token id 预测接下来的 token (最多 32 个)
    output = model_in_4bit_gpu.generate(input_ids,
                            max_new_tokens=32)

    # 将预测的 token id 解码为输出字符串
    output = output.cpu()
    output_str = tokenizer.decode(output[0], skip_special_tokens=True)
    
    print('-'*20, 'Output', '-'*20)
    print(output_str)
```

> **注意**
> 在英特尔 GPU 上运行优化的 LLM 刚开始的生成可能会比较慢。因此，建议在实际生成前进行一些**预热**的运行。
>
> 对于关于流式对话的下一节，我们可以将第 7.1.6 节中的这次生成视为一个预热。

## 6.1.6 流式对话

现在，让我们构建一个在英特尔 GPU 上运行的流式对话机器人，让 LLM 参与互动对话。聊天机器人的互动并没有什么魔法——它依然依赖于 LLM 预测以及生成下一个 token. 为了让 LLM 对话，我们需要将 prompt 适当的格式化为对话格式，例如：


```
<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant, who always answers as helpfully as possible, while being safe.
<</SYS>>

What is AI? [/INST]
```

此外，为了实现多轮对话，您需要将新的对话输入附加到之前的对话从而为模型制作一个新的 prompt，例如：

```
<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant, who always answers as helpfully as possible, while being safe.
<</SYS>>

What is AI? [/INST] AI is a term used to describe the development of computer systems that can perform tasks that typically require human intelligence, such as understanding natural language, recognizing images. </s><s> [INST] Is it dangerous? [INST]
```

这里我们展示了一个运行在 BigDL-LLM 优化过的 Llama 2 (7B) 模型上的支持流式显示的多轮对话实例。

首先，定义对话上下文格式[^1]，以便模型完成对话：

```python
SYSTEM_PROMPT = "You are a helpful, respectful and honest assistant, who always answers as helpfully as possible, while being safe."

def format_prompt(input_str, chat_history):
    prompt = [f'<s>[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n']
    do_strip = False
    for history_input, history_response in chat_history:
        history_input = history_input.strip() if do_strip else history_input
        do_strip = True
        prompt.append(f'{history_input} [/INST] {history_response.strip()} </s><s>[INST] ')
    input_str = input_str.strip() if do_strip else input_str
    prompt.append(f'{input_str} [/INST]')
    return ''.join(prompt)
```

[^1]: 对话上下文格式参考自[这里](https://huggingface.co/spaces/huggingface-projects/llama-2-7b-chat/blob/323df5680706d388eff048fba2f9c9493dfc0152/model.py#L20)以及[这里](https://huggingface.co/spaces/huggingface-projects/llama-2-7b-chat/blob/323df5680706d388eff048fba2f9c9493dfc0152/app.py#L9).

接下来，定义 `stream_chat` 函数，将模型输出持续添加到聊天记录中。这样可以确保对话上下文正确的被格式化从而便于下一次回复的生成。这里的响应是逐字生成的：

```python
from transformers import TextIteratorStreamer

def stream_chat(model, tokenizer, input_str, chat_history):
    # 通过聊天记录将对话上下文格式化为 prompt
    prompt = format_prompt(input_str, chat_history)
    input_ids = tokenizer([prompt], return_tensors='pt').to('xpu') # 为英特尔 GPU 指定 to('xpu')

    streamer = TextIteratorStreamer(tokenizer,
                                    skip_prompt=True, # 在生成的 token 中跳过 prompt
                                    skip_special_tokens=True)

    generate_kwargs = dict(
        input_ids,
        streamer=streamer,
        max_new_tokens=128
    )
    
    # 为了确保对生成文本的非阻塞访问，生成过程应在单独的线程中运行
    from threading import Thread
    
    thread = Thread(target=model.generate, kwargs=generate_kwargs)
    thread.start()

    output_str = []
    print("Response: ", end="")
    for stream_output in streamer:
        output_str.append(stream_output)
        print(stream_output, end="")

    # 将模型的输出添加至聊天记录中
    chat_history.append((input_str, ''.join(output_str)))
```

> **注意**
> 为了成功观察标准输出中的文本流行为，我们需要设置环境变量 `PYTHONUNBUFFERED=1` 以确保标准输出流直接发送到终端而不是先进行缓冲。
>
> [Hugging Face *transformers* streamer classes](https://huggingface.co/docs/transformers/main/generation_strategies#streaming) 目前还在开发中，未来可能会发生变化。

然后，我们可以通过允许连续的用户输入来实现人类和机器人之间的互动式、多轮流式对话：

```python
chat_history = []

print('-'*20, 'Stream Chat', '-'*20, end="")
while True:
    with torch.inference_mode():
        print("\n", end="")
        user_input = input("Input: ")
        if user_input == "stop": # 当用户输入 "stop" 时停止对话
            print("Stream Chat with Llama 2 (7B) stopped.")
            break
        stream_chat(model=model_in_4bit_gpu,
                    tokenizer=tokenizer,
                    input_str=user_input,
                    chat_history=chat_history)
```