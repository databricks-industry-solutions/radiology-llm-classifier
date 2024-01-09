# Databricks notebook source
# DBTITLE 1,Setup Notebook Parameters
dbutils.widgets.text("catalog", "hls_healthcare") #catalog, default value hls_healthcare
dbutils.widgets.text("volume", "hls_dev.radiology_llm") #volume name, default value hls_dev.radiology_llm

# COMMAND ----------

#TODO require cluster type === ??? 

# COMMAND ----------

#install libraries
!pip install -q accelerate==0.21.0 peft==0.4.0 bitsandbytes==0.40.2 transformers==4.31.0 trl==0.4.7

# COMMAND ----------

# MAGIC %sql
# MAGIC USE CATALOG ${catalog}

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE VOLUME IF NOT EXISTS ${volume}

# COMMAND ----------

#import libraries
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel, get_peft_model
from trl import SFTTrainer

# COMMAND ----------

#PEFT LORA configurations definitions used for multi-gpu
local_rank = -1
per_device_train_batch_size = 4
per_device_eval_batch_size = 4
gradient_accumulation_steps = 1
learning_rate = 2e-4
model_name = 'epfl-llm/meditron-7b'
max_grad_norm = 0.3 
weight_decay = 0.001
lora_alpha = 16
lora_dropout = 0.1
lora_r = 64
max_seq_length = None

# COMMAND ----------

use_4bit = True #enable QLORA
use_nested_quant = False 
bnb_4bit_compute_dtype = "float16"
bnb_4bit_quant_type = "nf4" #Quantization type (fp4 or nf4)
fp16 = False #Disable fp16 to False as we are using 4-bit precision QLORA
bf16 = False #Disable bf16 to False as we are using 4-bit precision QLORA
packing = False
gradient_checkpointing = True #Enable gradient checkpoint
optim = "paged_adamw_32bit" #Optimizer used for weight updates
lr_scheduler_type = "cosine" #The cosine lrs function has been shown to perform better than alternatives like simple linear annealing in practice.
max_steps = -1 #Number of optimizer update steps
warmup_ratio = 0.2 #Define training warmup fraction
group_by_length = True #Group sequences into batches with same length (saves memory and speeds up training considerably)
save_steps = 800 #Save checkpoint every X updates steps
logging_steps = 800 #Log every X updates steps
output_dir = "/Volumes/ang_nara_catalog/rad_llm/results"
device_map = {"": 0}

# COMMAND ----------

def load_model(model_name):
    """
    Function to load the LLM model weights, peft config, and tokenizer from HF
    """
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
    
    #Quantization config for QLORA
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,
    )
    #Model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        quantization_config=bnb_config
    )

    #Turn off cache to use the updated model params
    model.config.use_cache = False

    #This value is necessary to ensure exact reproducibility of the pretraining results
    model.config.pretraining_tp = 1
    
    #LORA Config
    peft_config = LoraConfig(
        lora_alpha=lora_alpha, #Controls LORA scaling; higher value makes the approximation more influencial
        lora_dropout=lora_dropout, #Probability that each neuron's output set to 0; prevents overfittig
        r=lora_r, #LORA rank param; lower value makes model faster but sacrifices performance
        bias="none",#For performance, we recommend setting bias to none first, and then lora_only, before trying all
        task_type="CAUSAL_LM", 
    )
    #Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer, peft_config

# COMMAND ----------

#TODO replace with secrets and require 
!huggingface-cli login --token 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

# COMMAND ----------

model, tokenizer, peft_config = load_model(model_name)

# COMMAND ----------

df = spark.sql("SELECT * FROM ang_nara_catalog.rad_llm.delta_rad_filtered")
df = df.toPandas()

# COMMAND ----------

#Generate list of dictionaries 
dataset_data = [
    {
        "instruction": "predict radiology labels for the clinical notes",
        "clinical_notes": row_dict["input"],
        "radiology_labels": row_dict["radiology_labels"]
    }
    for row_dict in df.to_dict(orient="records")
]

# COMMAND ----------

#Write dictionary list to a json file
import json
with open("/Volumes/ang_nara_catalog/rad_llm/clinical_data/filtered_clinical_notes.json", "w") as f:
   json.dump(dataset_data, f)

# COMMAND ----------

def format_rad(sample):
    """
    Function to create dataset as per Llama2 prompt format
    """
    instruction = f"<s>[INST] {sample['instruction']}"
    context = f"Here's some context: {sample['clinical_notes']}" if len(sample["clinical_notes"]) > 0 else None
    response = f" [/INST] {sample['radiology_labels']}"
    #Join all the parts together
    prompt = "".join([i for i in [instruction, context, response] if i is not None])
    return prompt

def template_dataset(sample):
    """
    Function to apply Llama2 prompt format on the entire dataset
    """
    sample["text"] = f"{format_rad(sample)}{tokenizer.eos_token}"
    return sample

#Apply prompt template per sample
dataset = load_dataset("json", data_files="/Volumes/ang_nara_catalog/rad_llm/clinical_data/filtered_clinical_notes.json", split="train")

# Shuffle the dataset
dataset_shuffled = dataset.shuffle(seed=42)
dataset = dataset.map(template_dataset, remove_columns=list(dataset.features))
dataset

# COMMAND ----------

#Model training 
training_arguments = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    ddp_find_unused_parameters=False,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=packing,
)

trainer.train()
trainer.model.save_pretrained(output_dir)

# COMMAND ----------

# Reload model in FP16 and merge it with LoRA weights
# To merge the model weights in HF, restart compute, run the first 6 cells, and run this cell

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map=device_map,
)
model = PeftModel.from_pretrained(base_model, output_dir)
model = model.merge_and_unload()

# Reload tokenizer to save it
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# COMMAND ----------

model.push_to_hub("RadiologyLLMs/RadLlama2-7b", use_auth_token=True, create_pr=1, max_shard_size='20GB')
tokenizer.push_to_hub("RadiologyLLMs/RadLlama2-7b", use_auth_token=True, create_pr=1)
