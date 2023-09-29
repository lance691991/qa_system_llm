from transformers import AutoModelForCausalLM
from peft import PeftModel
from transformers import AutoTokenizer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--base_model", type=str)
parser.add_argument("--peft_model", type=str)
parser.add_argument("--save_dir", type=str)
args = parser.parse_args()

# base_model_ckpt = "/home/ml/qa_system_llm/model/v2"
base_model = AutoModelForCausalLM.from_pretrained(args.base_model, trust_remote_code=True)
# peft_model = "/home/ml/qa_system_llm/model/Baichuan2-7B-Chat_v3"
tokenizer = AutoTokenizer.from_pretrained(args.peft_model, use_fast=False, trust_remote_code=True)
model = PeftModel.from_pretrained(base_model, args.peft_model)
merged_model = model.merge_and_unload(progressbar=True)
merged_model.save_pretrained(args.save_dir)
tokenizer.save_pretrained(args.save_dir)