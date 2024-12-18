# Preference Alignment

Aligning language models with human preferences. 
SFT helps models learn task, Preference alignment encourages outputs to match human expectations and values.

Typical alignment methods involves multiple stages:
1. SFT to adapt models to specific domains
2. Preference alignment (like RLHF or DPO) to improve response quality

Alternate approaches like ORPO combine instruction tuning and preference alignment


# 1. Direct Preference Optimization (DPO)
* DPO offers simplified approach to aligning language models with human preferences. 
* Unlike traditional RLHF methods that requires separate reward models and complex RL, DPO directly optimizes using the model using preference data.

### Understanding DPO
* DPO recasts preference alignment as classification problem on human preference data.

* Traditional RLHF approaches require training Reward model + complex RL algos like PPO to align model outputs.
`DPO` simplifies this process by `defining a loss function` that `optimizes the model's policy based on preferred vs non preferred` outputs.

* Used to train models like LLama

### How DPO works
* DPO proces requires SFT to adapt the model to the target domain.
* This creates a foundation for preference learning by training on standard instruction-following datasets.
* The model learns basic tasks completion while maintaining its general capabilities.

Next Preference Learning, 
* model is trained on pair of outputs - one preferred and one non-preferred.
* The prefered pair help the model understand which responses better align with human values and expectations.

* The core innovation of DPO lies in the direct optimization approach. 
* Rather than training a separate reward model, DPO uses a` binary cross-entropy loss` to directly update the model weight based on preference data.
* This streamlined process makes trainng more stable and efficient while achieving comparable or better results than traditional RLHF.

### DPO datasets
* Datasets for DPO are typically `created` by annotating `pair of preferred or non-preferred` response.

Below is example structure of single turn preference dataset for DPO
| Prompt        | Chosen        | Rejected  |
| ------------- |:-------------:|:---------:|
| ...           | ...           |   ...     |
| ...           | ...           |   ...     |
| ...           | ...           |   ...     |

Prompt Col - The prompt used to generate the `Chosen` and `Rejected` response.
`Chosen` and `Rejected` columns contains preferred and non-preferred response respectively.

### Implementation 
1. TRL - Transformers Reinforcement Learning Library makes implementation DPO straightforward.
DPOConfig and DPOTrainer classes follow same transformers style,

```python
from trl import DPOConfig, DPOTrainer

# Define arguments
training_args = DPOConfig(
    ...
)

# Initialize trainer
trainer = DPOTrainer(
    model,
    train_dataset=dataset,
    tokenizer=tokenizer,
    ...
)

# Train model
trainer.train()
```
### Best Practices
* Data quality is crucial for successful DPO implementation. 
* Preference dataset should include diverse examples covering different aspects of desired behavior. 
* Clear annotation guidelines ensures consistent labelling of preferred and non preferred response.
* Model performance == quality of your preference dataset.

* During training, monitor the loss convergence and validate performance on held-out data. 
* The beta parameters may need adjustments to balance preference learning with maintaining the model's general capabilities.
* Regular evaluation on diverse prompts helps ensure the model is learning the intended preferences without overfitting.

* Compare the model's outputs with the reference model to verify improvement in preference alignment.
* Testing on a variety of prompts, including edge cases, helps ensure robust preference learning across different scenarios.


example: https://huggingface.co/collections/argilla/preference-datasets-for-dpo-656f0ce6a00ad2dc33069478

# Resources:
1. https://argilla.io/blog/mantisnlp-rlhf-part-1/
2. https://arxiv.org/abs/2305.18290 DPO
3. https://arxiv.org/abs/2403.07691 ORPO
4. https://github.com/huggingface/alignment-handbook
5. https://github.com/huggingface/trl/blob/main/trl/scripts/dpo.py
