
# 7.1 Finetuning Llama 2 (7B) using QLoRA

To help you better understand the process of QLoRA Finetuning, in this tutorial, we provide a practical guide leveraging BigDL-LLM to tune a large language model to a specific task.  [Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf) is used as an example here to adapt the text generation implementation.

## 7.1.1 Enable BigDL-LLM on Intel GPUs

### 7.1.1.1 Install BigDL-LLM on Intel GPUs

After following the steps in [Readme](./README.md#70-environment-setup)  to set up the environment, you can install BigDL-LLM in terminal with the command below:
```bash
pip install --pre --upgrade bigdl-llm[xpu] -f https://developer.intel.com/ipex-whl-stable-xpu
pip install transformers==4.34.0
pip install peft==0.5.0
pip install accelerate==0.23.0
```

> **Note**
> If you didn't follow the installation guide above and are using an older version of `bigdl-llm` (specifically, older than `2.5.0b20240104`), you need to manually add `import intel_extension_for_pytorch as ipex` at the beginning of your code.

### 7.1.1.2 Set OneAPI Environment Variables

It is also necessary to set OneAPI environment variables for BigDL-LLM on Intel GPUs.

```bash
# configure OneAPI environment variables
source /opt/intel/oneapi/setvars.sh
```

## 7.1.2 QLoRA Finetuning

### 7.1.2.1 Load Model in Low Precision


A popular open-source LLM [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf) is chosen to illustrate the process of QLoRA Finetuning.

> **Note**
>
> You can specify the argument `pretrained_model_name_or_path` with both Huggingface repo id or local model path.
> If you have already downloaded the Llama 2 (7B) model, you could specify `pretrained_model_name_or_path` to the local model path.

With BigDL-LLM optimization, you can load the model with `bigdl.llm.transformers.AutoModelForCausalLM` instead of `transformers.AutoModelForCausalLM` to conduct implicit quantization.

For Intel GPUs, once you have the model in low precision, **set it to `to('xpu')`**.

```python
model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path = "meta-llama/Llama-2-7b-hf",
                                             load_in_low_bit="nf4",
                                             optimize_model=False,
                                             torch_dtype=torch.float16,
                                             modules_to_not_convert=["lm_head"])
model = model.to('xpu')

```

> **Note**
>
> We specify load_in_low_bit="nf4" here to apply 4-bit NormalFloat optimization. According to the [QLoRA paper](https://arxiv.org/pdf/2305.14314.pdf), using "nf4" could yield better model quality than "int4".

### 7.1.2.2 Prepare Model for Training
Then we apply `prepare_model_for_kbit_training` from `bigdl.llm.transformers.qlora` to preprocess the model for training. 

```python
from bigdl.llm.transformers.qlora import prepare_model_for_kbit_training
model.gradient_checkpointing_enable() # can further reduce memory but slower
model = prepare_model_for_kbit_training(model)
```

Next, we can obtain a PEFT model from the optimized model and a configuration object containing the parameters as follows:  

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
> **Note**
>
> Instead of `from peft import prepare_model_for_kbit_training, get_peft_model` as we did for regular QLoRA using bitandbytes and cuda, we import them from `bigdl.llm.transformers.qlora` here to get a BigDL-LLM compatible PEFT model. And the rest is just the same as regular LoRA finetuning process using `peft`.
>
> **Note**
>
> More explanation about `LoraConfig` parameters can be found in [Transformer LoRA Guides](https://huggingface.co/docs/peft/conceptual_guides/lora#common-lora-parameters-in-peft).
>

### 7.1.2.3 Load Dataset

A common dataset, [english quotes](https://huggingface.co/datasets/Abirate/english_quotes), is loaded to fine tune our model on famous quotes.
```python
from datasets import load_dataset
data = load_dataset("Abirate/english_quotes")
data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)
```

> **Note**
>
> The dataset path here is default to be Huggingface repo id. 
> If you have already downloaded the `.jsonl` file from [Abirate/english_quotes](https://huggingface.co/datasets/Abirate/english_quotes/blob/main/quotes.jsonl), you could use `data = load_dataset("json", data_files= "path/to/your/.jsonl/file")` to specify the local path instead of `data = load_dataset("Abirate/english_quotes")`.

### 7.1.2.4 Load Tokenizer
A tokenizer enables tokenizing and detokenizing process in LLM training and inference. You can use [Huggingface transformers](https://huggingface.co/docs/transformers/index) API to load the tokenizer directly. It can be used seamlessly with models loaded by BigDL-LLM. For Llama 2, the corresponding tokenizer class is `LlamaTokenizer`.

```python
from transformers import LlamaTokenizer
tokenizer = LlamaTokenizer.from_pretrained(pretrained_model_name_or_path="meta-llama/Llama-2-7b-chat-hf", trust_remote_code=True)
tokenizer.pad_token_id = 0
tokenizer.padding_side = "left"
```
> **Note**
>
> If you have already downloaded the Llama 2 (7B) model, you could specify `pretrained_model_name_or_path` to the local model path.

### 7.1.2.5 Run the Training

You can then start the training process by setting the `trainer` with existing tools on the HF ecosystem. Here we set `warmup_steps` to be 20 to accelerate the process of training.
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
        output_dir="outputs", # specify your own output path here
        optim="adamw_hf", # paged_adamw_8bit is not supported yet
        # gradient_checkpointing=True, # can further reduce memory but slower
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False  # silence the warnings, and we should re-enable it for inference
result = trainer.train()
```
We can get the following outputs showcasing our training loss:
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
The final LoRA weights and configurations have been saved to `${output_dir}/checkpoint-{max_steps}/adapter_model.bin` and `${output_dir}/checkpoint-{max_steps}/adapter_config.json`, which can be used for merging.

## 7.1.3 Merge the Model

After finetuning the model, you could merge the QLoRA weights back into the base model for export to Hugging Face format.

> **Note**
>
> Make sure your accelerate version is 0.23.0 to enable the merging process on CPU.

### 7.1.3.1 Load Pre-trained Model

```python
base_model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map={"": "cpu"},
    )
```

> **Note**
>
> In the merging state, load_in_low_bit="nf4" should be removed since we need to load the original model as the base model.



### 7.1.3.2 Merge the Weights



Then we can load the QLoRA weights to enable the merging process.

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
> **Note**
>
> Instead of `from peft import PeftModel`, we `import PeftModel from bigdl.llm.transformers.qlora` as a BigDL-LLM compatible model.
> 
> **Note**
> The adapter path is the local path you save the fine-tuned model, in our case is `./outputs/checkpoint-200`.
>

To verify if the LoRA weights have worked in conjunction with the pretrained model, the first layer weights (which in llama2 case are trainable queries) are extracted to highlight the difference.

```python
first_weight = base_model.model.layers[0].self_attn.q_proj.weight
first_weight_old = first_weight.clone()
lora_weight = lora_model.base_model.model.model.layers[0].self_attn.q_proj.weight
assert torch.allclose(first_weight_old, first_weight)
```
With the new merging method `merge_and_unload`, we can easily combine the fine-tuned model with pre-trained model, and testify whether the weights have changed with the `assert` statement. 

```python
lora_model = lora_model.merge_and_unload()
lora_model.train(False)
assert not torch.allclose(first_weight_old, first_weight)
```
You may get the outputs below without error report to indicate the successful conversion.
```
Using pad_token, but it is not set yet.
Using pad_token, but it is not set yet.
```
Finally we can save the fine-tuned model in a specified local path (in our case is `./outputs/checkpoint-200-merged`).
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


## 7.1.4 Inference with Fine-tuned model

After merging and deploying the models, we can test the performance of the fine-tuned model. 
The detailed instructions of running LLM inference with BigDL-LLM optimizations could be found in [Chapter 6](../ch_6_GPU_Acceleration/6_1_GPU_Llama2-7B.md), here we quickly go through the preparation of model inference.

### 7.1.4.1 Inference with the Fine-tuned Model

```python
model_path = "./outputs/checkpoint-200-merged"
model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path = model_path,load_in_4bit=True)
model = model.to('xpu')
tokenizer = LlamaTokenizer.from_pretrained(pretrained_model_name_or_path = model_path)
```
> **Note**
> The `model_path` argument should be consistent with the output path of your merged model.
>
Then we can verify if the fine-tuned model can produce reasonable and philosophical response with the new dataset added.
```python
with torch.inference_mode():
    input_ids = tokenizer.encode('The paradox of time and eternity is', 
                                  return_tensors="pt").to('xpu')
    output = model.generate(input_ids, max_new_tokens=32)
    output = output.cpu()
    output_str = tokenizer.decode(output[0], skip_special_tokens=True)
    print(output_str)
```

We can repeat the process with the pre-trained model by replacing the `model_path` argument to verify the improvement after finetuning process. Now we can compare the answer of the pre-trained Model with the fine-tuned one:

> **Pre-trained Model**
```
The paradox of time and eternity is that time is not eternal, but eternity is. nobody knows how long time is.
The paradox of time and eternity is
```
> **Fine-tuned Model**
```
The paradox of time and eternity is that, on the one hand, we experience time as linear and progressive, and on the other hand, we experience time as cyclical. And the
```

We can see the result shares the same style and context with the samples contained in the fine-tuned Dataset. And note that we only trained the Model for some epochs in a few minutes based on the optimization of BigDL-LLM.

Here are more results with same prompts input for pretrained and fine-tuned models:

|   ♣ Pre-trained Model   | ♣ Fine-tuned Model  |
|         -----          |       -----         |
|   **There are two things that matter:** Einzelnes and the individual. Everyone has heard of the "individual," but few have heard of the "individuum," or "   |  **There are two things that matter:** the quality of our relationships and the legacy we leave. And I think that all of us as human beings are searching for it, no matter where |
|   **In the quiet embrace of the night,** I felt the earth move. Unterscheidung von Wörtern und Ausdrücken.  |  **In the quiet embrace of the night,** the world is still and the stars are bright. My eyes are closed, my heart is at peace, my mind is at rest. I am ready for  |



