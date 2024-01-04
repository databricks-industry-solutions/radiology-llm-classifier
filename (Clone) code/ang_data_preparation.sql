-- Databricks notebook source
-- MAGIC %md
-- MAGIC ## Preprocessing

-- COMMAND ----------

-- MAGIC %python
-- MAGIC csv_path = "/Volumes/uc_demos_angelina_leigh/nara_ang_demo/nara_ang/clinical_notes_dataset.csv"

-- COMMAND ----------

use catalog uc_demos_angelina_leigh;
use schema nara_ang_demo

-- COMMAND ----------

CREATE OR REPLACE TEMPORARY VIEW clinical_notes
USING csv
OPTIONS (
  'path' '/Volumes/uc_demos_angelina_leigh/nara_ang_demo/nara_ang/clinical_notes_dataset.csv',
  'header' 'true',      
  'inferSchema' 'true'  
);

-- COMMAND ----------

SELECT 
  protocol_labels,
  COUNT(*) AS protocol_counts
FROM 
  clinical_notes
WHERE 
  protocol_labels IS NOT NULL
GROUP BY 
  protocol_labels
HAVING 
  COUNT(*) >= 3
--ORDER BY 
  --COUNT(*) DESC

-- COMMAND ----------

describe extended clinical_notes

-- COMMAND ----------

drop table if exists clinical_notes_table;
CREATE TABLE clinical_notes_table
as
select * 
from clinical_notes

-- COMMAND ----------

select * from clinical_notes_table limit 10

-- COMMAND ----------

select 
protocol_labels,
count(*)
from clinical_notes_table
group by protocol_labels

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Setup

-- COMMAND ----------

-- MAGIC %python
-- MAGIC !pip install -U pip
-- MAGIC !pip install accelerate==0.18.0
-- MAGIC !pip install appdirs==1.4.4
-- MAGIC !pip install bitsandbytes==0.37.2
-- MAGIC !pip install datasets==2.10.1
-- MAGIC !pip install fire==0.5.0
-- MAGIC !pip install git+https://github.com/huggingface/peft.git
-- MAGIC !pip install git+https://github.com/huggingface/transformers.git
-- MAGIC !pip install torch==2.0.0
-- MAGIC !pip install sentencepiece==0.1.97
-- MAGIC !pip install tensorboardX==2.6
-- MAGIC !pip install gradio==3.23.0

-- COMMAND ----------

-- MAGIC %python
-- MAGIC !pip install --upgrade transformers

-- COMMAND ----------

-- MAGIC %python
-- MAGIC pip install --upgrade accelerate

-- COMMAND ----------

-- MAGIC %python
-- MAGIC dbutils.library.restartPython()

-- COMMAND ----------

-- MAGIC %python
-- MAGIC import transformers
-- MAGIC import textwrap
-- MAGIC from transformers import LlamaTokenizer, LlamaForCausalLM
-- MAGIC import os
-- MAGIC import sys
-- MAGIC from typing import List
-- MAGIC
-- MAGIC from peft import (
-- MAGIC     LoraConfig,
-- MAGIC     get_peft_model,
-- MAGIC     get_peft_model_state_dict,
-- MAGIC     prepare_model_for_int8_training,
-- MAGIC )
-- MAGIC
-- MAGIC import fire
-- MAGIC import torch
-- MAGIC from datasets import load_dataset
-- MAGIC import pandas as pd
-- MAGIC
-- MAGIC import matplotlib.pyplot as plt
-- MAGIC import matplotlib as mpl
-- MAGIC import seaborn as sns
-- MAGIC from pylab import rcParams
-- MAGIC import json
-- MAGIC
-- MAGIC %matplotlib inline
-- MAGIC sns.set(rc={'figure.figsize':(8, 6)})
-- MAGIC sns.set(rc={'figure.dpi':100})
-- MAGIC sns.set(style='white', palette='muted', font_scale=1.2)

-- COMMAND ----------

-- MAGIC %python
-- MAGIC !gdown 1xQ89cpZCnafsW5T3G3ZQWvR7q682t2BN

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Alpaca LoRa

-- COMMAND ----------

-- MAGIC %python
-- MAGIC BASE_MODEL = "decapoda-research/llama-7b-hf"
-- MAGIC
-- MAGIC model = LlamaForCausalLM.from_pretrained(
-- MAGIC     BASE_MODEL,
-- MAGIC     load_in_8bit=True,
-- MAGIC     torch_dtype=torch.float16,
-- MAGIC     device_map="auto",
-- MAGIC )
-- MAGIC
-- MAGIC tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL)
-- MAGIC
-- MAGIC tokenizer.pad_token_id = (
-- MAGIC     0  # unk. we want this to be different from the eos token
-- MAGIC )
-- MAGIC tokenizer.padding_side = "left"

-- COMMAND ----------

select * from uc_demos_angelina_leigh.nara_ang_demo.clinical_notes_table

-- COMMAND ----------

-- MAGIC %python
-- MAGIC df = spark.sql("SELECT * FROM uc_demos_angelina_leigh.nara_ang_demo.clinical_notes_table")

-- COMMAND ----------

-- MAGIC %python
-- MAGIC df = df.toPandas()

-- COMMAND ----------

-- MAGIC %python
-- MAGIC df.head()

-- COMMAND ----------

-- MAGIC %python
-- MAGIC dataset_data = [
-- MAGIC     {
-- MAGIC         "instruction": "predict protocol labels for clinical impressions",
-- MAGIC         "input": row_dict["reasonofstudy"],
-- MAGIC         "output": row_dict["protocol_labels"]
-- MAGIC     }
-- MAGIC     for row_dict in df.to_dict(orient="records")
-- MAGIC ]

-- COMMAND ----------

CREATE OR REPLACE TEMPORARY VIEW clinical_notes_json
USING json
OPTIONS (
  'path' '/Volumes/uc_demos_angelina_leigh/nara_ang_demo/nara_ang/clinical_notes_dataset.json',
  'header' 'true',      
  'inferSchema' 'true'  
);

-- COMMAND ----------

drop table if exists clinical_notes_table_train;
CREATE TABLE clinical_notes_table_train
as
select * 
from clinical_notes_json

-- COMMAND ----------

select * from clinical_notes_table_train

-- COMMAND ----------

-- MAGIC %python
-- MAGIC df_train = spark.sql("SELECT * FROM uc_demos_angelina_leigh.nara_ang_demo.clinical_notes_table_train")
-- MAGIC df_train = df_train.toPandas()

-- COMMAND ----------

-- MAGIC %python
-- MAGIC CUTOFF_LEN = 256

-- COMMAND ----------

-- MAGIC %python
-- MAGIC def generate_prompt(data_point):
-- MAGIC     return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.  # noqa: E501
-- MAGIC ### Instruction:
-- MAGIC {data_point["instruction"]}
-- MAGIC ### Input:
-- MAGIC {data_point["input"]}
-- MAGIC ### Response:
-- MAGIC {data_point["output"]}"""

-- COMMAND ----------

-- MAGIC %python
-- MAGIC def tokenize(prompt, add_eos_token=True):
-- MAGIC     # there's probably a way to do this with the tokenizer settings
-- MAGIC     # but again, gotta move fast
-- MAGIC     result = tokenizer(
-- MAGIC         prompt,
-- MAGIC         truncation=True,
-- MAGIC         max_length=CUTOFF_LEN,
-- MAGIC         padding=False,
-- MAGIC         return_tensors=None,
-- MAGIC     )
-- MAGIC     if (
-- MAGIC         result["input_ids"][-1] != tokenizer.eos_token_id
-- MAGIC         and len(result["input_ids"]) < CUTOFF_LEN
-- MAGIC         and add_eos_token
-- MAGIC     ):
-- MAGIC         result["input_ids"].append(tokenizer.eos_token_id)
-- MAGIC         result["attention_mask"].append(1)
-- MAGIC
-- MAGIC     result["labels"] = result["input_ids"].copy()
-- MAGIC
-- MAGIC     return result
-- MAGIC
-- MAGIC def generate_and_tokenize_prompt(data_point):
-- MAGIC     full_prompt = generate_prompt(data_point)
-- MAGIC     tokenized_full_prompt = tokenize(full_prompt)
-- MAGIC     return tokenized_full_prompt

-- COMMAND ----------

-- MAGIC %python
-- MAGIC df_train["train"]

-- COMMAND ----------

-- MAGIC %python
-- MAGIC train_val = df_train["train"].train_test_split(
-- MAGIC     test_size=200, shuffle=True, seed=42
-- MAGIC )
-- MAGIC train_data = (
-- MAGIC     train_val["train"].shuffle().map(generate_and_tokenize_prompt)
-- MAGIC )
-- MAGIC val_data = (
-- MAGIC     train_val["test"].shuffle().map(generate_and_tokenize_prompt)
-- MAGIC )

-- COMMAND ----------

-- MAGIC %python
-- MAGIC LORA_R = 8
-- MAGIC LORA_ALPHA = 16
-- MAGIC LORA_DROPOUT= 0.05
-- MAGIC LORA_TARGET_MODULES = [
-- MAGIC     "q_proj",
-- MAGIC     "v_proj",
-- MAGIC ]
-- MAGIC
-- MAGIC BATCH_SIZE = 128
-- MAGIC MICRO_BATCH_SIZE = 4
-- MAGIC GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
-- MAGIC LEARNING_RATE = 3e-4
-- MAGIC TRAIN_STEPS = 300
-- MAGIC OUTPUT_DIR = "experiments"

-- COMMAND ----------

-- MAGIC %python
-- MAGIC model = prepare_model_for_int8_training(model)
-- MAGIC config = LoraConfig(
-- MAGIC     r=LORA_R,
-- MAGIC     lora_alpha=LORA_ALPHA,
-- MAGIC     target_modules=LORA_TARGET_MODULES,
-- MAGIC     lora_dropout=LORA_DROPOUT,
-- MAGIC     bias="none",
-- MAGIC     task_type="CAUSAL_LM",
-- MAGIC )
-- MAGIC model = get_peft_model(model, config)
-- MAGIC model.print_trainable_parameters()

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Training

-- COMMAND ----------

-- MAGIC %python
-- MAGIC training_arguments = transformers.TrainingArguments(
-- MAGIC     per_device_train_batch_size=MICRO_BATCH_SIZE,
-- MAGIC     gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
-- MAGIC     warmup_steps=100,
-- MAGIC     max_steps=TRAIN_STEPS,
-- MAGIC     learning_rate=LEARNING_RATE,
-- MAGIC     fp16=True,
-- MAGIC     logging_steps=10,
-- MAGIC     optim="adamw_torch",
-- MAGIC     evaluation_strategy="steps",
-- MAGIC     save_strategy="steps",
-- MAGIC     eval_steps=50,
-- MAGIC     save_steps=50,
-- MAGIC     output_dir=OUTPUT_DIR,
-- MAGIC     save_total_limit=3,
-- MAGIC     load_best_model_at_end=True,
-- MAGIC     report_to="tensorboard"
-- MAGIC )

-- COMMAND ----------

-- MAGIC %python
-- MAGIC data_collator = transformers.DataCollatorForSeq2Seq(
-- MAGIC     tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
-- MAGIC )

-- COMMAND ----------

-- MAGIC %python
-- MAGIC trainer = transformers.Trainer(
-- MAGIC     model=model,
-- MAGIC     train_dataset=train_data,
-- MAGIC     eval_dataset=val_data,
-- MAGIC     args=training_arguments,
-- MAGIC     data_collator=data_collator
-- MAGIC )
-- MAGIC model.config.use_cache = False
-- MAGIC old_state_dict = model.state_dict
-- MAGIC model.state_dict = (
-- MAGIC     lambda self, *_, **__: get_peft_model_state_dict(
-- MAGIC         self, old_state_dict()
-- MAGIC     )
-- MAGIC ).__get__(model, type(model))
-- MAGIC
-- MAGIC model = torch.compile(model)
-- MAGIC
-- MAGIC trainer.train()
-- MAGIC model.save_pretrained(OUTPUT_DIR)

-- COMMAND ----------

-- MAGIC %python
-- MAGIC %load_ext tensorboard
-- MAGIC %tensorboard --logdir experiments/runs

-- COMMAND ----------

-- MAGIC %python
-- MAGIC from huggingface_hub import notebook_login
-- MAGIC
-- MAGIC notebook_login()

-- COMMAND ----------

-- MAGIC %python
-- MAGIC model.push_to_hub("curiousily/alpaca-bitcoin-tweets-sentiment", use_auth_token=True)

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Inference

-- COMMAND ----------

-- MAGIC %python
-- MAGIC !git clone https://github.com/tloen/alpaca-lora.git
-- MAGIC %cd alpaca-lora
-- MAGIC !git checkout a48d947

-- COMMAND ----------

-- MAGIC %python
-- MAGIC !python generate.py \
-- MAGIC     --load_8bit \
-- MAGIC     --base_model 'decapoda-research/llama-7b-hf' \
-- MAGIC     --lora_weights 'curiousily/alpaca-bitcoin-tweets-sentiment' \
-- MAGIC     --share_gradio

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## References
-- MAGIC  - [Bitcoin Sentiment Dataset](https://www.kaggle.com/datasets/aisolutions353/btc-tweets-sentiment)
