# Parameter-Efficient Fine-tuning (PEFT)
* As language model grow larger, traditional fine-tuning becomes increasingly challenging.
* FFT of 1.7B parameter model requires substantial GPU memory, makes storing separate model copies expensive, and risks catastrophic forgetting of the model's original capabilities.
* PEFT method address this challenge by modifying only a small subset of model params while keeping most of the model frozen.

PEFT methods introduce approaches to adapt models using fewer trainable parameters - often < 1% of the model size. This dramatic reduction in trainable parameters enables:
1. Fine-tuning on consumer hardware with limited GPU memory.
2. Storing multiple task specific adaptations efficiently. 
3. Better generalization in low- data scenerios. 
4. Faster training and iterations cycles.

# Avaliable methods

### 1. LoRA (Low-Rank Adaptation)
Instead of modifying the entire model, **LoRA injects trainable matrices into the model's attention layers**. This approach typically reduces trainable parameters by about 90% while maintaining comaparable perf to full-funing. 

LoRA most widely adopted PEFT method. It works by **adding small rank decomposition matrices to the attention weights**, typically reducing trainable parameters by about 90%

### Understanding LoRA
**LoRA** (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique that freezes the pre-trained model weights and injects trainable rank decomposition matrices into the model's layers. LoRA **decomposes the weights updates into small matrices through low-rank decomposition**, significantly reducing the number of trainable parameters while maintaining model perf. 
* eg for GPT-3 175B, LoRA reduced trainable parameters by 10000x and GPU memory requirements by 3x compared to full fine-tuning.

LoRA works by adding a pair of rank decomposition matrices to transformer layer, typically focusing on attention weights. During inference, these adapter weights can be merged with the base model, resulting in no additional latency overhead. 

### Loading LoRA Adapters
Adapters can be **loaded** onto a pretrained model with **load_adaptors()**, which is useful for trying out different adaptors whose weights aren't merged. **Set** the active adaptors weights with the **set_adapters()** funtion. To **return the base model**, you could use to **unload()** to unload all the LoRA modules.

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("<base_model_name>")
peft_model_id = "<peft_adapter_id>"
model = PeftModel.from_pretrained(base_model, peft_model_id)
```
### Merging LoRA Adapters
After training with LoRA, you might want to merge the adapter weights back into the base model for easier deployment, elimnating the need to load adapters seperately during inference. The merging process requires attention to memory management and precision. Maintain consistent precision(e.g., float16) of both the base model and adapter weights throughout the process. Before deploying, always validate the merged model by comparing its outputs and performance metrics with the adapter-based version. 

Adapters are also be convenient for switching between different tasks or domains. you can load the base model and adapter weights separately. This allows the quick switching between different task-specific weights.

### Using TRL with PEFT
PEFT methods can be combined with TRL for efficient fine-tuning. This integration is particularly useful for RLHF as it reduces memory requirements.


```python
from peft import LoraConfig
from transformers import AutoModelForCausalLM

# Load model with PEFT config
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Load model on specific device
model = AutoModelForCausalLM.from_pretrained(
    "your-model-name",
    load_in_8bit=True,  # Optional: use 8-bit precision
    device_map="auto",
    peft_config=lora_config
)
```



### Basic Merging Implementation

```python
import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel

# 1. Load the base model
base_model = AutoModelForCausalLM.from_pretrained(
    "base_model_name",
    torch_dtype=torch.float16,
    device_map="auto"
)

# 2. Load the PEFT model with adapter
peft_model = PeftModel.from_pretrained(
    base_model,
    "path/to/adapter",
    torch_dtype=torch.float16
)

# 3. Merge adapter weights with base model
try:
    merged_model = peft_model.merge_and_unload()
except RuntimeError as e:
    print(f"Merging failed: {e}")
    # Implement fallback strategy or memory optimization

# 4. Save the merged model
merged_model.save_pretrained("path/to/save/merged_model")
```



### 2. Prompt Tuning
PT offers even lighter approach by **adding trainable tokens to the input** rather than modifying model weights. Prompt tuning is less popular than LoRA, but can be useful technique for quickly adaptiong a model to new tasks or domains.

# Resources:
1. https://huggingface.co/docs/peft/index
2. https://arxiv.org/abs/2106.09685
3. https://arxiv.org/abs/2305.14314 
4. https://huggingface.co/blog/peft
5. https://www.philschmid.de/fine-tune-llms-in-2024-with-trl
