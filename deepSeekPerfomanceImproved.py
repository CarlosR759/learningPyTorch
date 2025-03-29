from transformers import  AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    #device_map="auto",
    offload_state_dict=True,
)

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")

user_input = input("Insert your query: ")
inputs = f"<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"
encoded_inputs = tokenizer(inputs, return_tensors="pt").to(0)

with torch.no_grad():
    outputs = model(**encoded_inputs)

token_id = outputs.logits[0][-1].argmax()
answer = tokenizer.decode([token_id])

print(answer)
