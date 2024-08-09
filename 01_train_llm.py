# Databricks notebook source
# MAGIC %md # Setup Configs and Libraries

# COMMAND ----------

# DBTITLE 1,Setup Notebook Parameters
dbutils.widgets.text("catalog", "hls_healthcare") 
dbutils.widgets.text("database", "hls_dev")
dbutils.widgets.text("volume_storage", "radiology_reslts")

dbutils.widgets.text("model_name", "epfl-llm/meditron-7b") 
dbutils.widgets.text("hugging-face-token-secret", "medtron-hf-token")

# COMMAND ----------

llm_volume = (dbutils.widgets.get("catalog") + 
                    "." + dbutils.widgets.get("database") +
                    "." + dbutils.widgets.get("volume_storage"))

llm_volume_output = ( "/Volumes/" + 
                     dbutils.widgets.get("catalog") +
                     "/" + dbutils.widgets.get("database") +
                     "/" + dbutils.widgets.get("volume_storage") )


# COMMAND ----------

# DBTITLE 1,External Library Dependencies 
#install libraries
!pip install -q accelerate==0.21.0 peft==0.4.0 bitsandbytes==0.40.2 transformers==4.31.0 trl==0.4.7 guardrail-ml==0.0.12
!pip install -q unstructured["local-inference"]==0.7.4 pillow
!pip install pydantic==1.8.2 
!pip install -q transformers[deepspeed] mpi4py

# COMMAND ----------

# DBTITLE 1,Set catalog / database location
# MAGIC %sql
# MAGIC CREATE CATALOG IF NOT EXISTS ${catalog};
# MAGIC USE CATALOG ${catalog};
# MAGIC CREATE DATABASE IF NOT EXISTS ${database};
# MAGIC USE ${database};
# MAGIC CREATE VOLUME IF NOT EXISTS ${llm_volume};

# COMMAND ----------

# DBTITLE 1,Import libraries 
#import libraries
import os, json, torch
from datasets import Dataset, NamedSplit
from pyspark.sql.functions import *
from pyspark.sql import Row
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
from guardrail.client import run_metrics
from accelerate.utils import DistributedType

# COMMAND ----------

# DBTITLE 1,Set training configs
#PEFT LORA configurations definitions used for multi-gpu
local_rank = -1
per_device_train_batch_size = 20
per_device_eval_batch_size = 20
gradient_accumulation_steps = 1
learning_rate = 2e-4
model_name = dbutils.widgets.get("model_name")
max_grad_norm = 0.3 
weight_decay = 0.001
lora_alpha = 16
lora_dropout = 0.1
lora_r = 64
max_seq_length = None
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
save_steps = 500 #Save checkpoint every X updates steps
logging_steps = 500 #Log every X updates steps
eval_steps= 500 #Eval steps
evaluation_strategy = "steps" #Display val loss for every step
save_strategy = "steps" 
output_dir = llm_volume_output
device_map={"":torch.cuda.current_device()}
training_data_tablename = dbutils.widgets.get("training_data_tablename")

# COMMAND ----------

# MAGIC %md # Load Training Data
# MAGIC Note, this is synthetic data for training purposes and has restricted use, including the resulting fine tuned model.

# COMMAND ----------

#load handwritten notes from csv file in Github repo
#
def load_data(path="/data/30k_handwritten_clinical_notes.csv"):
  return (spark.read.format("csv")
          .option("header",True)
          .load("file:///" + os.getcwd() +  path)
  )
#
# Only use where there exists 50 or more labels
#   @param default function used to load data from Github repo
#     - can provide your own load_data() function, or your own filtered_table() function to source data from 
# 
#  @returns dataframe with columns "input", "radiology_labels", and "instruction"
#
def filtered_table(df = load_data):
  df().createOrReplaceTempView("radiology_data")
  return spark.sql("""
    SELECT t.input, t.radiology_labels
      FROM (
         SELECT t.*, COUNT(*) OVER (PARTITION BY radiology_labels) AS cnt
         FROM radiology_data t
      ) t
      WHERE cnt > 50
  """).withColumn("instruction", lit('predict radiology label for the clinical notes')) 

# COMMAND ----------

# DBTITLE 1,Show sample training data
df = filtered_table()
df.show()

# COMMAND ----------

# MAGIC %md # Model Training Execution

# COMMAND ----------

# MAGIC %md ## Fine Tuning GPU Options
# MAGIC Runtimes: 
# MAGIC T4 GPU (Driver: g4dn.16xlarge, Worker: g4dn.16xlarge) = ~5 hours  
# MAGIC NVIDIA A10G on G5 GPU (Driver: g5.24xlarge, Worker: g5.24xlarge) = ~2 hours

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
        quantization_config=bnb_config,
        use_auth_token=True
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
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_auth_token=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer, peft_config

# COMMAND ----------

# DBTITLE 1,Login to Hugging Face
val = dbutils.secrets.get(scope=dbutils.widgets.get("hugging-face-token-secret"), key="token")
!huggingface-cli login --token $val

# COMMAND ----------

#Download model 
model, tokenizer, peft_config = load_model(model_name)

# COMMAND ----------

# DBTITLE 1,Prompts for training
def format_rad(sample):
    """
    Function to create dataset as per Llama2 prompt format
    """
    instruction = f"<s>[INST] {sample['instruction']}"
    context = f"Here's some context: {sample['input']}" if len(sample["input"]) > 0 else None
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

# Shuffle the dataset
dataset = Dataset.from_pandas(df.toPandas())
dataset_shuffled = dataset.shuffle(seed=42)

# COMMAND ----------

#Split dataset into train and val
train_val = dataset.train_test_split(
    test_size=8000, shuffle=True, seed=42
)
train_data = (
    train_val["train"].map(template_dataset)
)
val_data = (
    train_val["test"].map(template_dataset)
)

# COMMAND ----------

#Model training 
training_arguments = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    evaluation_strategy = evaluation_strategy,
    save_strategy = save_strategy,
    save_steps=save_steps,
    eval_steps=eval_steps,
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
    train_dataset = train_data,
    eval_dataset = val_data,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=packing,
)
trainer.train()
trainer.model.save_pretrained(output_dir)
