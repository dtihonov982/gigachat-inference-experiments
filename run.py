import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

model_name = "ai-sage/GigaChat3.1-10B-A1.8B"
cache_dir = Path(__file__).parent / "cache"

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    cache_dir=cache_dir,
)
model.generation_config = GenerationConfig.from_pretrained(model_name, cache_dir=cache_dir)

messages = [{"role": "user", "content": "Hello! What can you do?"}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}

outputs = model.generate(**inputs, max_new_tokens=512)
prompt_len = inputs["input_ids"].shape[1]
result = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)
print(result)
