# from utils.create_dataset import lm_datasets, model_checkpoint
from transformers import AutoModelForCausalLM
from transformers import Trainer, TrainingArguments, HfArgumentParser
from peft import LoraConfig, TaskType, get_peft_model
from datasets import Dataset
import json
from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling

data_dic = json.load(open("/home/ml/qa_system_llm/copus/json_files/regulation_allall.json"))
model_checkpoint = "/home/ml/qa_system_llm/model/Baichuan2-7B-Chat"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=False, trust_remote_code=True)
block_size, overlap_size = 128, 64
total_len = 0
for i in range(len(data_dic["text"])):
    total_len += len(data_dic["text"][i])
    if total_len >= block_size:
        data_dic["text"][i] += tokenizer.eos_token
        total_len = 0
ds = Dataset.from_dict(data_dic)
def tokenize_function(examples):
    return tokenizer(examples["text"])
tokenized_datasets = ds.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])
def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, overlap_size)]
        for k, t in concatenated_examples.items()
    }
    return result
lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    batch_size=1000,
    num_proc=4,
)


# tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

model = AutoModelForCausalLM.from_pretrained(model_checkpoint, trust_remote_code=True)
peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=["W_pack"],
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
        )
model.enable_input_require_grads()
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
parser = HfArgumentParser(TrainingArguments)
training_args = parser.parse_args_into_dataclasses()[0]

trainer = Trainer(
        model=model, args=training_args, train_dataset=lm_datasets, data_collator=data_collator, tokenizer=tokenizer
    )
trainer.train()
trainer.save_state()
trainer.save_model(output_dir=training_args.output_dir)

