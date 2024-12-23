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



### 2. Prompt Tuning
PT offers even lighter approach by **adding trainable tokens to the input** rather than modifying model weights. Prompt tuning is less popular than LoRA, but can be useful technique for quickly adaptiong a model to new tasks or domains.

# Resources:
1. https://huggingface.co/docs/peft/index
2. https://arxiv.org/abs/2106.09685
3. https://arxiv.org/abs/2305.14314 
4. https://huggingface.co/blog/peft
5. https://www.philschmid.de/fine-tune-llms-in-2024-with-trl
