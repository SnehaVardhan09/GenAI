* Chat template Essential for structuring interactions between language models and users.
* Provide consistent format for conversations, ensuring that models understand the context and role of each message while maintaining appropriate response pattern.

## BaseModels vs Instruct Models
* Base model is trained on raw text to predict next token, `SmolLM2-135M`
* Instruct model is fine tuned specifically to follow instructions and engage in conversation. `SmolLM2-135M-Instruct`

-> for converting base to instruct, we need to format our prompts in consistent way that the model understands. 
-> differnet chat templates comes in, each chat template for differnt usecase.

## Understanding
Formatting the conversations when communicating with Language models.
* Including System-level instructions, user messages, and assistant responses in structured format, helps us maintain consistency

<|im_start|>user
Hi there!<|im_end|>
<|im_start|>assistant
Nice to meet you!<|im_end|>
<|im_start|>user
Can I ask a question?<|im_end|>
<|im_start|>assistant

* The transformers library will take care of the chat templates for you in relation to model's tokenizer. 
* All we have to do is structure our messagses in the correct way and the tokenizer will take care of the rest.

message = [
    {"role": "system", "content": " "},
    {"role": "user", "content": " "},
    {"role": "assistant", "content" " "}
]

System Messages: set the foundation for how the model should behave. They act as persistent instructions that influence all subsequent interactions.

system_message = {
    "role": "system",
    "content": "You are a professional customer service agent. Always be polite, clear, and helpful."
}