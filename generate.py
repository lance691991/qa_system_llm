from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers.generation.utils import GenerationConfig

tokenizer = AutoTokenizer.from_pretrained("/home/ml/qa_system_llm/model/v1", use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("/home/ml/qa_system_llm/model/v1", device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
model.generation_config = GenerationConfig.from_pretrained("/home/ml/qa_system_llm/model/v1")
messages = []
messages.append({"role": "user", "content": "归档信息包可用性"})
response = model.chat(tokenizer, messages)
print(response)