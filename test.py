import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
tokenizer = AutoTokenizer.from_pretrained("/home/ml/qa_system_llm/model/Baichuan-13B-Chat_v4", use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("/home/ml/qa_system_llm/model/Baichuan-13B-Chat_v4", device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)
model.generation_config = GenerationConfig.from_pretrained("/home/ml/qa_system_llm/model/Baichuan-13B-Chat_v4")
messages = []
messages.append({"role": "user", "content": "问：什么是声像档案"})
response = model.chat(tokenizer, messages)
print(response)