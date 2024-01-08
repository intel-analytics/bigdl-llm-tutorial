# 6.1 Run Llama 2 (7B) on Intel GPUs

You can use BigDL-LLM to load any Hugging Face *transformers* model for acceleration on Intel GPUs. With BigDL-LLM, PyTorch models (in FP16/BF16/FP32) hosted on Hugging Face can be loaded and optimized automatically on Intel GPUs with low-bit quantizations (supported precisions include INT4/NF4/INT5/INT8).

In this tutorial, you will learn how to run LLMs on Intel GPUs with BigDL-LLM optimizations, and based on that build a stream chatbot. A popular open-source LLM [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) is used as an example.

## 6.1.1 Install BigDL-LLM on Intel GPUs

First of all, install BigDL-LLM in your prepared environment. For best practices of environment setup on Intel GPUs, refer to the [README](./README.md#70-environment-setup) in this chapter.

In terminal, run:

```bash
pip install bigdl-llm[xpu] -f https://developer.intel.com/ipex-whl-stable-xpu
```

> **Note**
> If you didn't follow the installation guide above and are using an older version of `bigdl-llm` (specifically, older than `2.5.0b20240104`), you need to manually add `import intel_extension_for_pytorch as ipex` at the beginning of your code.

It is also required to set oneAPI environment variables for BigDL-LLM on Intel GPUs.

```bash
# configure oneAPI environment variables
source /opt/intel/oneapi/setvars.sh
```

## 6.1.2 (Optional) Download Llama 2 (7B)

To download the [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) model from Hugging Face, you will need to obtain access granted by Meta. Please follow the instructions provided [here](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf/tree/main) to request access to the model.

After receiving the access, download the model with your Hugging Face token:

```python
from huggingface_hub import snapshot_download

model_path = snapshot_download(repo_id='meta-llama/Llama-2-7b-chat-hf',
                               token='hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX') # change it to your own Hugging Face access token
```

> **Note**
> The model will by default be downloaded to `HF_HOME='~/.cache/huggingface'`.

## 6.1.3 Load Model in Low Precision

One common use case is to load a Hugging Face *transformers* model in low precision, i.e. conduct **implicit** quantization while loading.

For Llama 2 (7B), you could simply import `bigdl.llm.transformers.AutoModelForCausalLM` instead of `transformers.AutoModelForCausalLM`, and specify `load_in_4bit=True` or `load_in_low_bit` parameter accordingly in the `from_pretrained` function.

For Intel GPUs, **once you have the model in low precision, set it to `to('xpu')`.**

**For INT4 Optimizations (with `load_in_4bit=True`):**

```python
from bigdl.llm.transformers import AutoModelForCausalLM

model_in_4bit = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path="meta-llama/Llama-2-7b-chat-hf",
                                                     load_in_4bit=True)
model_in_4bit_gpu = model_in_4bit.to('xpu')
```

> **Note**
> BigDL-LLM has supported `AutoModel`, `AutoModelForCausalLM`, `AutoModelForSpeechSeq2Seq` and `AutoModelForSeq2SeqLM`.
>
> If you have already downloaded the Llama 2 (7B) model and skipped step [7.1.2.2](#712-optional-download-llama-2-7b), you could specify `pretrained_model_name_or_path` to the model path.

**(Optional) For INT8 Optimizations (with `load_in_low_bit="sym_int8"`):**

```python
# note that the AutoModelForCausalLM here is imported from bigdl.llm.transformers
model_in_8bit = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path="meta-llama/Llama-2-7b-chat-hf",
    load_in_low_bit="sym_int8"
)
model_in_8bit_gpu = model_in_8bit.to('xpu')
```

> **Note**
> * Currently, `load_in_low_bit` supports options `'sym_int4'`, `'asym_int4'`, `'sym_int5'`, `'asym_int5'` or `'sym_int8'`, in which 'sym' and 'asym' differentiate between symmetric and asymmetric quantization. Option `'nf4'` is also supported, referring to 4-bit NormalFloat. Floating point precisions `'fp4'`, `'fp8'`, `'fp16'` and mixed precisions including `'mixed_fp4'` and `'mixed_fp8'` are also supported.
>
> * `load_in_4bit=True` is equivalent to `load_in_low_bit='sym_int4'`.

## 6.1.4 Load Tokenizer 

A tokenizer is also needed for LLM inference. You can use [Huggingface transformers](https://huggingface.co/docs/transformers/index) API to load the tokenizer directly. It can be used seamlessly with models loaded by BigDL-LLM. For Llama 2, the corresponding tokenizer class is `LlamaTokenizer`.

```python
from transformers import LlamaTokenizer

tokenizer = LlamaTokenizer.from_pretrained(pretrained_model_name_or_path="meta-llama/Llama-2-7b-chat-hf")
```

> **Note**
> If you have already downloaded the Llama 2 (7B) model and skipped step [7.1.2.2](#712-optional-download-llama-2-7b), you could specify `pretrained_model_name_or_path` to the model path.

## 6.1.5 Run Model

You can then do model inference with BigDL-LLM optimizations on Intel GPUs almostly the same way as using official `transformers` API. **The only difference is to set `to('xpu')` for token ids**. A Q&A dialog template is created for the model to complete.

```python
import torch

with torch.inference_mode():
    prompt = 'Q: What is CPU?\nA:'
    
    # tokenize the input prompt from string to token ids;
    # with .to('xpu') specifically for inference on Intel GPUs
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to('xpu')

    # predict the next tokens (maximum 32) based on the input token ids
    output = model_in_4bit_gpu.generate(input_ids,
                            max_new_tokens=32)

    # decode the predicted token ids to output string
    output = output.cpu()
    output_str = tokenizer.decode(output[0], skip_special_tokens=True)
    
    print('-'*20, 'Output', '-'*20)
    print(output_str)
```

> **Note**
> The initial generation of optimized LLMs on Intel GPUs could be slow. Therefore, it's advisable to perform a **warm-up** run before the actual generation.
>
> For the next section of stream chat, we could treat this time of generation in section 7.1.6 as a warm-up.

## 6.1.6 Stream Chat

Now, let's build a stream chatbot that runs on Intel GPUs, allowing LLMs to engage in interactive conversations. Chatbot interaction is no magic - it still relies on the prediction and generation of next tokens by LLMs. To make LLMs chat, we need to properly format the prompts into a converation format, for example:

```
<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant, who always answers as helpfully as possible, while being safe.
<</SYS>>

What is AI? [/INST]
```

Further, to enable a multi-turn chat experience, you need to append the new dialog input to the previous conversation to make a new prompt for the model, for example: 

```
<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant, who always answers as helpfully as possible, while being safe.
<</SYS>>

What is AI? [/INST] AI is a term used to describe the development of computer systems that can perform tasks that typically require human intelligence, such as understanding natural language, recognizing images. </s><s> [INST] Is it dangerous? [INST]
```

Here we show a multi-turn chat example with stream capability on BigDL-LLM optimized Llama 2 (7B) model. 

First, define the conversation context format[^1] for the model to complete:

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

[^1]: The conversation context format is referenced from [here](https://huggingface.co/spaces/huggingface-projects/llama-2-7b-chat/blob/323df5680706d388eff048fba2f9c9493dfc0152/model.py#L20) and [here](https://huggingface.co/spaces/huggingface-projects/llama-2-7b-chat/blob/323df5680706d388eff048fba2f9c9493dfc0152/app.py#L9).

Next, define the `stream_chat` function, which continuously adds model outputs to the chat history. This ensures that conversation context can be properly formatted for next generation of responses. Here, the response is generated in a streaming (word-by-word) way:

```python
from transformers import TextIteratorStreamer

def stream_chat(model, tokenizer, input_str, chat_history):
    # format conversation context as prompt through chat history
    prompt = format_prompt(input_str, chat_history)
    input_ids = tokenizer([prompt], return_tensors='pt').to('xpu') # specify to('xpu') for Intel GPUs

    streamer = TextIteratorStreamer(tokenizer,
                                    skip_prompt=True, # skip prompt in the generated tokens
                                    skip_special_tokens=True)

    generate_kwargs = dict(
        input_ids,
        streamer=streamer,
        max_new_tokens=128
    )
    
    # to ensure non-blocking access to the generated text, generation process should be ran in a separate thread
    from threading import Thread
    
    thread = Thread(target=model.generate, kwargs=generate_kwargs)
    thread.start()

    output_str = []
    print("Response: ", end="")
    for stream_output in streamer:
        output_str.append(stream_output)
        print(stream_output, end="")

    # add model output to the chat history
    chat_history.append((input_str, ''.join(output_str)))
```

> **Note**
> To successfully observe the text streaming behavior in standard output, we need to set the environment variable `PYTHONUNBUFFERED=1` to ensure that the standard output streams are directly sent to the terminal without being buffered first.
>
> The [Hugging Face *transformers* streamer classes](https://huggingface.co/docs/transformers/main/generation_strategies#streaming) is currently being developed and is subject to future changes.

We can then achieve interactive, multi-turn stream chat between humans and the bot by allowing continuous user input:

```python
chat_history = []

print('-'*20, 'Stream Chat', '-'*20, end="")
while True:
    with torch.inference_mode():
        print("\n", end="")
        user_input = input("Input: ")
        if user_input == "stop": # let's stop the conversation when user input "stop"
            print("Stream Chat with Llama 2 (7B) stopped.")
            break
        stream_chat(model=model_in_4bit_gpu,
                    tokenizer=tokenizer,
                    input_str=user_input,
                    chat_history=chat_history)
```