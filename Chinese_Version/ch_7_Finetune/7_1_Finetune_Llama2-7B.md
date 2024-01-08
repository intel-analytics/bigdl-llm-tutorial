# 7.1 使用 QLoRA 微调 Llama 2 (7B)

为了帮助您更好地理解 QLoRA 微调过程，在本教程中，我们提供了一个实用指南，利用 BigDL-LLM 将大语言模型对特定的下游任务进行微调。 这里使用 [Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf) 作为示例来完成文本生成任务。

## 7.1.1 在 Intel GPU 上启用 BigDL-LLM

### 7.1.1.1 安装 BigDL-LLM

按照[Readme](./README.md#70-environment-setup)中的步骤设置环境后，您可以使用以下命令在终端中安装 BigDL-LLM 以及相应的依赖环境：

```bash
pip install --pre --upgrade bigdl-llm[xpu] -f https://developer.intel.com/ipex-whl-stable-xpu
pip install transformers==4.34.0
pip install peft==0.5.0
pip install accelerate==0.23.0
```

> **注意**
> 如果您没有使用上面的安装方式，而是使用了旧版本的`bigdl-llm`(早于`2.5.0b20240104`版本)，您需要在代码开头手动导入`import intel_extension_for_pytorch as ipex`。

### 7.1.1.2 配置 oneAPI 环境变量

完成安装后，您需要为英特尔 GPU 配置 oneAPI 环境变量。

```bash
# 配置 oneAPI 环境变量
source /opt/intel/oneapi/setvars.sh
```

## 7.1.2 QLoRA 微调

### 7.1.2.1 以低精度加载模型

本教程选择流行的开源 LLM 模型 [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf)来说明 QLoRA 模型微调的过程。

> **注意**
>
> 您可以使用 Huggingface 库 id 或本地模型路径来指定参数 `pretrained_model_name_or_path`。
> 如果您已经下载了 Llama 2 (7B) 模型，您可以将 `pretrained_model_name_or_path` 指定为本地模型路径。

通过 BigDL-LLM 优化，您可以使用`bigdl.llm.transformers.AutoModelForCausalLM`替代`transformers.AutoModelForCausalLM`来加载模型来进行隐式量化。

对于英特尔 GPU，您应在`from_pretrained`函数中特别设置 `optimize_model=False`。一旦获得低精度模型，请将其设置为`to('xpu')`。

```python
model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path = "meta-llama/Llama-2-7b-hf",
                                             load_in_low_bit="nf4",
                                             optimize_model=False,
                                             torch_dtype=torch.float16,
                                             modules_to_not_convert=["lm_head"])
model = model.to('xpu')
```
> **注意**
>
> 我们指定 load_in_low_bit="nf4" 以应用 4 位 NormalFloat 优化。 根据 [QLoRA 论文](https://arxiv.org/pdf/2305.14314.pdf)，使用 “nf4” 量化比 “int4” 能达到更优的模型效果。

### 7.1.2.2 准备训练模型

我们可以应用`bigdl.llm.transformers.qlora`中的`prepare_model_for_kbit_training`对模型进行预处理，以进行训练的准备。

```python
from bigdl.llm.transformers.qlora import prepare_model_for_kbit_training
model.gradient_checkpointing_enable() #可以进一步减少内存使用但速度较慢
model = prepare_model_for_kbit_training(model)
```

接下来，我们可以从预处理后的模型中创建一个 PEFT 模型并配置它的参数，如下所示：

```python
from bigdl.llm.transformers.qlora import get_peft_model
from peft import LoraConfig

config = LoraConfig(r=8, 
                    lora_alpha=32, 
                    target_modules=["q_proj", "k_proj", "v_proj"], 
                    lora_dropout=0.05, 
                    bias="none", 
                    task_type="CAUSAL_LM")
model = get_peft_model(model, config)

```
> **注意**
>
> 我们从`bigdl.llm.transformers.qlora`导入与BigDL-LLM兼容的 PEFT 模型，替代原先使用 bitandbytes 库和 cuda 的`from peft importprepare_model_for_kbit_training, get_peft_model`用法。其使用方法和用`peft`库进行 QLoRA 微调方法相同。
> 
> **注意**
>
> 有关 LoraConfig 参数的更多说明可以在 [Transformer LoRA 指南](https://huggingface.co/docs/peft/conceptual_guides/lora#common-lora-parameters-in-peft)中查看。

### 7.1.2.3 加载数据集

我们加载通用数据集 [english quotes](https://huggingface.co/datasets/Abirate/english_quotes) 来根据英语名言来微调我们的模型。

```python
from datasets import load_dataset
data = load_dataset("Abirate/english_quotes")
data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)
```

> **注意**
>
> 如果您已经从 [Abirate/english_quotes](https://huggingface.co/datasets/Abirate/english_quotes/blob/main/quotes.jsonl) 下载了 `.jsonl` 文件，您可以使用 `data = load_dataset( "json", data_files= "path/to/your/.jsonl/file")` 指定本地路径，以替代从 huggingface repo id 的加载方法 `data = load_dataset("Abirate/english_quotes")`。

### 7.1.2.4 加载Tokenizer

分词器可以在 LLM 训练和推理中实现分词和去分词过程。您可以使用 [Huggingface Transformers](https://huggingface.co/docs/transformers/index) API来加载 LLM 推理需要的分词器，它可以与 BigDL-LLM 加载的模型无缝配合使用。对于Llama 2，对应的tokenizer类为`LlamaTokenizer`。

```python
from transformers import LlamaTokenizer
tokenizer = LlamaTokenizer.from_pretrained(pretrained_model_name_or_path="meta-llama/Llama-2-7b-chat-hf", trust_remote_code=True)
tokenizer.pad_token_id = 0
tokenizer.padding_side = "left"
```

> **注意**
>
> 如果您已经下载了 Llama 2 (7B) 模型，您可以将 `pretrained_model_name_or_path` 指定为本地模型路径。

### 7.1.2.5 进行训练

现在您可以通过使用 HuggingFace 系统设置`trainer`来开始训练过程。 这里我们将`warmup_steps`设置为 20 以加速训练过程。
```python
import transformers
trainer = transformers.Trainer(
    model=model,
    train_dataset=data["train"],
    args=transformers.TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps= 1,
        warmup_steps=20,
        max_steps=200,
        learning_rate=2e-4,
        save_steps=100,
        fp16=True,
        logging_steps=20,
        output_dir="outputs", # 这里指定你自己的输出路径
        optim="adamw_hf", # 尚不支持 paged_adamw_8bit优化
        # gradient_checkpointing=True, # #可以进一步减少内存使用但速度较慢
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False  # 消除警告，进行推理时应重新启用
result = trainer.train()
```
我们可以获得以下输出来展示我们的训练损失：
```
/home/arda/anaconda3/envs/yining-llm-qlora/lib/python3.9/site-packages/transformers/optimization.py:411: FutureWarning: This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this warning
  warnings.warn(
{'loss': 1.7193, 'learning_rate': 0.0002, 'epoch': 0.03}                                                             
{'loss': 1.3242, 'learning_rate': 0.00017777777777777779, 'epoch': 0.06}                                             
{'loss': 1.2266, 'learning_rate': 0.00015555555555555556, 'epoch': 0.1}                                              
{'loss': 1.1534, 'learning_rate': 0.00013333333333333334, 'epoch': 0.13}                                             
{'loss': 0.9368, 'learning_rate': 0.00011111111111111112, 'epoch': 0.16}                                             
{'loss': 0.9321, 'learning_rate': 8.888888888888889e-05, 'epoch': 0.19}                                              
{'loss': 0.9902, 'learning_rate': 6.666666666666667e-05, 'epoch': 0.22}                                              
{'loss': 0.8593, 'learning_rate': 4.4444444444444447e-05, 'epoch': 0.26}                                             
{'loss': 1.0055, 'learning_rate': 2.2222222222222223e-05, 'epoch': 0.29}                                             
{'loss': 1.0081, 'learning_rate': 0.0, 'epoch': 0.32}                                                                
{'train_runtime': xxx, 'train_samples_per_second': xxx, 'train_steps_per_second': xxx, 'train_loss': 1.1155566596984863, 'epoch': 0.32}
100%|██████████████████████████████████████████████████████████████████████████████| 200/200 [xx:xx<xx:xx,  xxxs/it]
```

最终的 LoRA 模型权重和配置文件会保存到`${output_dir}/checkpoint-{max_steps}/adapter_model.bin`和`${output_dir}/checkpoint-{max_steps}/adapter_config.json`。

## 7.1.3 合并模型

微调模型后，您可以将QLoRA模型权重和基本模型合并并导出为Hugging Face格式。

> **注意**
>
> 请确保您的`accelerate`库版本是 0.23.0，以便在CPU上启用模型合并。

### 7.1.3.1 加载预训练模型

```python
base_model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map={"": "cpu"},
    )
```

> **注意**
>
> 在合并状态下，应删除 load_in_low_bit="nf4"，因为我们需要加载原始模型作为基本模型。

### 7.1.3.2 合并权重

然后我们可以加载训练出的LoRA权重来准备合并。

```python
from bigdl.llm.transformers.qlora import PeftModel
adapter_path = "./outputs/checkpoint-200"
lora_model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        device_map={"": "cpu"},
        torch_dtype=torch.float16,
    )
```

> **注意**
>
> 我们用`import PeftModel from bigdl.llm.transformers.qlora`替代`from peft import PeftModel`来导入 BigDL-LLM 兼容的 PEFT 模型。

> **注意**
> 
> 适配器路径是您保存微调模型的本地路径，在本例中是`./outputs/checkpoint-200`。

为了验证LoRA权重和预训练模型权重有效合并，我们提取第一层权重（在 llama2模型中为attention中的queries）来对比其差异。

```python
first_weight = base_model.model.layers[0].self_attn.q_proj.weight
first_weight_old = first_weight.clone()
lora_weight = lora_model.base_model.model.model.layers[0].self_attn.q_proj.weight
assert torch.allclose(first_weight_old, first_weight)
```
通过`merge_and_unlaod`方法可以将微调后的模型与预训练的模型进行合并，并通过`assert`声明来验证权重是否发生变化。

```python
lora_model = lora_model.merge_and_unload()
lora_model.train(False)
assert not torch.allclose(first_weight_old, first_weight)
```
如果返回一下输出并没有报错，则说明模型合并成功。

```
Using pad_token, but it is not set yet.
Using pad_token, but it is not set yet.
```

最后，我们可以将合并的模型保存在指定的本地路径中（在我们的例子中是`./outputs/checkpoint-200-merged`）。

```python
output_path = ./outputs/checkpoint-200-merged
lora_model_sd = lora_model.state_dict()
deloreanized_sd = {
        k.replace("base_model.model.", ""): v
        for k, v in lora_model_sd.items()
        if "lora" not in k
    }
base_model.save_pretrained(output_path, state_dict=deloreanized_sd)
tokenizer.save_pretrained(output_path)

```

## 7.1.4 使用微调模型进行推理

合并和部署模型后，我们可以测试微调模型的性能。
使用BigDL-LLM优化的推理过程详细说明可以在[第6章](../ch_6_GPU_Acceleration/6_1_GPU_Llama2-7B.md)中找到，这里我们快速完成模型推理的准备工作。

### 7.1.4.1 使用微调模型进行推理

```python
model_path = "./outputs/checkpoint-200-merged"
model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path = model_path,load_in_4bit=True)
model = model.to('xpu')
tokenizer = LlamaTokenizer.from_pretrained(pretrained_model_name_or_path = model_path)
```

> **注意**
> `model_path` 参数应该与合并模型的输出路径一致。

我们可以验证微调模型在增添新的数据集训练后是否能做出富有哲理的回答。

```python
with torch.inference_mode():
    input_ids = tokenizer.encode('The paradox of time and eternity is', 
                                  return_tensors="pt").to('xpu')
    output = model.generate(input_ids, max_new_tokens=32)
    output = output.cpu()
    output_str = tokenizer.decode(output[0], skip_special_tokens=True)
    print(output_str)
```

我们可以仅替换`model_path`参数来使用预训练模型重复该过程，将预训练模型的答案与微调模型的答案进行比较：

> **预训练模型**
```
The paradox of time and eternity is that time is not eternal, but eternity is. nobody knows how long time is.
The paradox of time and eternity is
```
> **微调模型**
```
The paradox of time and eternity is that, on the one hand, we experience time as linear and progressive, and on the other hand, we experience time as cyclical. And the
```

我们可以看到微调模型的推理结果与新增数据集中有相同的词汇以及近似的文本风格。基于 BigDL-LLM 的优化，我们可以在仅仅几分钟的训练之内达到这个效果。

以下是更多预训练模型和微调模型的对比结果：

|   ♣ 预训练模型   | ♣ 微调模型  |
|         -----          |       -----         |
|   **There are two things that matter:** Einzelnes and the individual. Everyone has heard of the "individual," but few have heard of the "individuum," or "   |  **There are two things that matter:** the quality of our relationships and the legacy we leave.And I think that all of us as human beings are searching for it, no matter where |
|   **In the quiet embrace of the night,** I felt the earth move. Unterscheidung von Wörtern und Ausdrücken.  |  **In the quiet embrace of the night,** the world is still and the stars are bright. My eyes are closed, my heart is at peace, my mind is at rest. I am ready for  |