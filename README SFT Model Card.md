---
tags:
- axolot
- code
- coding
- Aguila
- axolot
model-index:
- name: edumunozsala/aguila-7b-instructft-bactrian-x
  results: []
license: apache-2.0
language:
- code
datasets:
- MBZUAI/Bactrian-X
pipeline_tag: text-generation
---


# Aguila 7B SFT model on spanish language 👩‍💻 

**Aguila 7B** supervised instruction finetuned on the [Bactrian-X dataset](https://github.com/mbzuai-nlp/Bactrian-X) by using the **Axolot** library in 4-bit with [PEFT](https://github.com/huggingface/peft) library.

## Pretrained description

[Aguila-7B](projecte-aina/aguila-7b)

Ǎguila-7B is a transformer-based causal language model for Catalan, Spanish, and English. It is based on the Falcon-7B model and has been trained on a 26B token trilingual corpus collected from publicly available corpora and crawlers.

More information available in the following post from Medium.com: [Introducing Ǎguila, a new open-source LLM for Spanish and Catalan](https://medium.com/@mpamies247/introducing-a%CC%8Cguila-a-new-open-source-llm-for-spanish-and-catalan-ee1ebc70bc79)


## Training data

[MBZUAI/Bactrian-X](https://huggingface.co/datasets/MBZUAI/Bactrian-X)

The Bactrain-X dataset is a collection of 3.4M instruction-response pairs in 52 languages, that are obtained by translating 67K English instructions (alpaca-52k + dolly-15k) into 51 languages using Google Translate API. The translated instructions are then fed to ChatGPT (gpt-3.5-turbo) to obtain its natural responses, resulting in 3.4M instruction-response pairs in 52 languages (52 languages x 67k instances = 3.4M instances).

Here we only use the spanish split of the dataset.

### Training hyperparameters

The following `axolot` configuration was used during training:
```
base_model: projecte-aina/aguila-7b
# required by falcon custom model code: https://huggingface.co/tiiuae/falcon-7b/tree/main
trust_remote_code: true
model_type: AutoModelForCausalLM
tokenizer_type: AutoTokenizer
is_falcon_derived_model: true
load_in_8bit: false
# enable 4bit for QLoRA
load_in_4bit: true
gptq: false
strict: false

push_dataset_to_hub:
datasets:
  - path: edumunozsala/Bactrian-X-es-50k
    type: alpaca
dataset_prepared_path:
val_set_size: 0.05
# enable QLoRA
adapter: qlora
lora_model_dir:
sequence_len: 2048
max_packed_sequence_len:

# hyperparameters from QLoRA paper Appendix B.2
# "We find hyperparameters to be largely robust across datasets"
lora_r: 64
lora_alpha: 16
# 0.1 for models up to 13B
# 0.05 for 33B and 65B models
lora_dropout: 0.05
# add LoRA modules on all linear layers of the base model
lora_target_modules:
lora_target_linear: true
lora_fan_in_fan_out:

wandb_project:
wandb_entity:
wandb_watch:
wandb_name:
wandb_log_model:

output_dir: ./qlora-out

# QLoRA paper Table 9
# - 16 for 7b & 13b
# - 32 for 33b, 64 for 64b
# Max size tested on A6000
# - 7b: 40
# - 40b: 4
# decrease if OOM, increase for max VRAM utilization
micro_batch_size: 4
gradient_accumulation_steps: 2
num_epochs: 2
# Optimizer for QLoRA
optimizer: paged_adamw_32bit
torchdistx_path:
lr_scheduler: cosine
# QLoRA paper Table 9
# - 2e-4 for 7b & 13b
# - 1e-4 for 33b & 64b
learning_rate: 0.0002
train_on_inputs: false
group_by_length: false
bf16: auto
fp16:
tf32: true
gradient_checkpointing: true
# stop training after this many evaluation losses have increased in a row
# https://huggingface.co/transformers/v4.2.2/_modules/transformers/trainer_callback.html#EarlyStoppingCallback
# early_stopping_patience: 3
resume_from_checkpoint:
auto_resume_from_checkpoints: true
local_rank:
logging_steps: 200
xformers_attention: true
flash_attention:
gptq_groupsize:
gptq_model_v1:
warmup_steps: 10
evals_per_epoch: 1
saves_per_epoch: 1
debug:
deepspeed:
weight_decay: 0.000001
fsdp:
fsdp_config:
special_tokens:
  pad_token: "<|endoftext|>"
  bos_token: "<|endoftext|>"
  eos_token: "<|endoftext|>"
```

### Framework versions
- torch=="2.1.2"
- flash-attn=="2.5.0"
- deepspeed=="0.13.1"
- axolotl=="0.4.0"

### Example of usage

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "edumunozsala/aguila-7b-instructft-bactrian-x"

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(model_id, load_in_4bit=True, torch_dtype=torch.float16, 
                                             device_map="auto", trust_remote_code=True)

instruction="Piense en una solución para reducir la congestión del tráfico."

input=""

prompt = f"""### Instrucción:
{instruction}

### Entrada:
{input}

### Respuesta:
"""

input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
# with torch.inference_mode():
outputs = model.generate(input_ids=input_ids, max_new_tokens=256, do_sample=True, top_p=0.9,temperature=0.3)

print(f"Prompt:\n{prompt}\n")
print(f"Generated instruction:\n{tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]}")

```

### Citation

```
@misc {edumunozsala_2023,
	author       = { {Eduardo Muñoz} },
	title        = { aguila-7b-instructft-bactrian-x },
	year         = 2024,
	url          = { https://huggingface.co/edumunozsala/aguila-7b-instructft-bactrian-x },
	publisher    = { Hugging Face }
}
```