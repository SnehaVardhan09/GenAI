# Parameter-Efficient Fine-tuning (PEFT)
* As language model grow larger, traditional fine-tuning becomes increasingly challenging.
* FFT of 1.7B parameter model requires substantial GPU memory, makes storing separate model copies expensive, and risks catastrophic forgetting of the model's original capabilities.
* PEFT method address this challenge by modifying only a small subset of model params while keeping most of the model frozen,



