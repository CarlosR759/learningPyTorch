from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

#Prints true to see if cuda or ROCm is enabled.
print(torch.cuda.is_available())


tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")

#This commented line is for using quantization with the model. You should add the bitsandbytes library and not use an already quantized model. Not tested btw.
#Device_map Automatically detects and load the model to best gpu device or default cpu.
'''model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                                               load_in_4bit=True,
                                               device_map="auto",)
'''

# Define the input text
#We need to send the prompt with ChatLM syntax.
user_input = input("Insert your query: ")
input_text = f"<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"
# Tokenize the input text
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

# Generate text
with torch.no_grad():
    outputs = model.generate(**inputs,
                            max_length=32768, #You can short this value to get fastter but smaller answers, sometimes not totally completed.
                            temperature=0.7,
                            do_sample=True,)


# Decode the output
#generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
# Print the generated text
print(generated_text)
