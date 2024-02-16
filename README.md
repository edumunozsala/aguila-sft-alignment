# 👩‍💻 From an Aguila 7B pretrained model to a SFT version and to a Reinforcement Learning Aligment 

## Aguila 7B

**Aguila 7B** supervised instruction finetuned on the [Bactrian-X dataset](https://github.com/mbzuai-nlp/Bactrian-X) by using the **Axolot** library in 4-bit with [PEFT](https://github.com/huggingface/peft) library.

Ǎguila is a 7B parameters LLM that has been trained on a mixture of Spanish, Catalan and English data, adding up to a total of 26B tokens.
It uses the Falcon-7b model as a starting point, a state-of-the-art English language model that was openly released just a few months ago by the Technology Innovation Institute.

The Aguila project explores the possibility of using an English LLM as a starting point to train a model in a different language. In particular, they adapt the Falcon-7B model to two additional languages, namely Spanish and Catalan, by swapping the tokenizer and adjusting the embedding layer. The first step for a successful language adaptation is the replacement of the model’s tokenizer. This is crucial because using the original English-based tokenizer would lead to a very high token-to-word ratio. Thus, a new BPE tokenizer was trained on a mixture of Spanish, Catalan and English data. Secondly, the embedding layer is modified by keeping only the weights that correspond to shared tokens (those present both in the old and new tokenizer) and replacing the rest with the overall mean value.

You can find more information in the following post from Medium.com: [Introducing Ǎguila, a new open-source LLM for Spanish and Catalan](https://medium.com/@mpamies247/introducing-a%CC%8Cguila-a-new-open-source-llm-for-spanish-and-catalan-ee1ebc70bc79)


## Build a Supervised FineTuned version

The instruction finetuned model is available on the [Huggingface Hub](https://huggingface.co/edumunozsala/aguila-7b-instructft-bactrian-x): edumunozsala/aguila-7b-instructft-bactrian-x 

Large language model (LLM) fine-tuning is the process of taking pre-trained models and further training them on smaller, specific datasets to refine their capabilities and improve performance in a particular task or domain. Fine-tuning is about turning general-purpose models and turning them into specialized models. Supervised fine-tuning means updating a pre-trained language model using labeled data to do a specific task. This is different from unsupervised methods, where data isn't checked. Usually, the initial training of the language model is unsupervised, but fine-tuning is supervised. In our case, we apply an instruction tuning approach.

One approach to improve a model's performance on various tasks is instruction fine-tuning. It's about training the machine learning model using examples that guide the model to respond to an instruction or the query. For preparing the training data, there are many open-source datasets that offer insights into user behaviors and preferences, even if they aren't directly formatted as instructional data. You can take a common dataset and turn it into instruction prompt datasets for fine-tuning. Prompt template libraries include many templates for different tasks and different datasets. The aim is to adapt the previously learned general knowledge to the nuances and specific patterns present in the new dataset, thereby making the model more specialized and effective for the target task.

*"The main difference between instruction tuning and standard supervised fine-tuning lies in the data that the model is trained on. Whereas supervised fine-tuning trains models on input examples and their corresponding outputs, instruction tuning augments input-output examples with instructions, which enables instruction-tuned models to generalize more easily to new tasks."*

Fromn Sebastian Ruder in the post [Instruction Tuning Vol. 1](https://newsletter.ruder.io/p/instruction-tuning-vol-1).

Axolot is the library we use to finetune our base model, you can configure all your parameters in a YAML file, including LoRA or QLoRA, and run yor training with a single command.

## Axolotl

Axolotl is a tool designed to streamline the fine-tuning of various AI models, offering support for multiple configurations and architectures.

Features:

- Train various Huggingface models such as llama, pythia, falcon, mpt
- Supports fullfinetune, lora, qlora, relora, and gptq
- Customize configurations using a simple yaml file or CLI overwrite
- Load different dataset formats, use custom formats, or bring your own tokenized datasets
- Integrated with xformer, flash attention, rope scaling, and multipacking
- Works with single GPU or multiple GPUs via FSDP or Deepspeed
- Easily run with Docker locally or on the cloud
- Log results and optionally checkpoints to wandb or mlflow

## The dataset for SFT

[MBZUAI/Bactrian-X](https://huggingface.co/datasets/MBZUAI/Bactrian-X)

The Bactrain-X dataset is a collection of 3.4M instruction-response pairs in 52 languages, that are obtained by translating 67K English instructions (alpaca-52k + dolly-15k) into 51 languages using Google Translate API. The translated instructions are then fed to ChatGPT (gpt-3.5-turbo) to obtain its natural responses, resulting in 3.4M instruction-response pairs in 52 languages (52 languages x 67k instances = 3.4M instances).

Here we only use the spanish split of the dataset and we take a sample of 50,000 instructions.

## Aligment Learning 

### Still in progress 

## Content
**Still In progress**

- Instruction tuning notebook `Aguila-7B-Instruction-tuned-Axolot.ipynb`: 

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

## Contributing
If you find some bug or typo, please let me know or fixit and push it to be analyzed. 

## License

Copyright 2023 Eduardo Muñoz

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
