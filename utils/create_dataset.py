from datasets import Dataset
import json
from transformers import AutoTokenizer

data_dic = json.load(open("/home/ml/qa_system_llm/copus/json_files/regulation_all.json"))
ds = Dataset.from_dict(data_dic)
model_checkpoint = "/home/ml/qa_system_llm/model/Baichuan2-7B-Chat"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=False, trust_remote_code=True)
def tokenize_function(examples):
    return tokenizer(examples["text"])
tokenized_datasets = ds.map(tokenize_function, batched=True, num_proc=2, remove_columns=["text"])
block_size = 128
def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    return result
lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    batch_size=1000,
    num_proc=4,
)

# print(tokenizer.decode(lm_datasets[0]["input_ids"]))
