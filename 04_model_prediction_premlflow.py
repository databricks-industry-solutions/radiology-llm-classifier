# Databricks notebook source
#install libraries
!pip install -q accelerate==0.21.0 peft==0.4.0 bitsandbytes==0.40.2 transformers==4.31.0 trl==0.4.7

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

!huggingface-cli login --token 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

# COMMAND ----------

import os
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("RadiologyLLMs/RadLlama2-7b")
model = AutoModelForCausalLM.from_pretrained("RadiologyLLMs/RadLlama2-7b")

# COMMAND ----------

import re
def trim_llm_output(text):
    # Define a regular expression to match any punctuation
    punctuation_regex = re.compile(r'[.,;!?:()<]+')

    # Find the first occurrence of punctuation
    match = punctuation_regex.search(text)

    if match:
        # Split the text at the first occurrence of punctuation
        split_text = text[:match.end()-1]
        return split_text
    else:
        # No punctuation found, return the original text
        return text

# COMMAND ----------

from transformers import pipeline
def pred_wrapper(model, tokenizer, prompt, model_id=1, show_metrics=True, temp=0.1, max_length=1):
    pipe = pipeline(task="text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_length=max_length,
                    do_sample=True,
                    temperature=temp)

    pipe = pipeline(task="text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_length=100)
    result = pipe(f"<s>[INST] {prompt} [/INST]")
    generated_text = result[0]['generated_text']

    # Find the index of "### Assistant" in the generated text
    index = generated_text.find("[/INST] ")
    if index != -1:
        # Extract the substring after "### Assistant"
        substring_after_assistant = generated_text[index + len("[/INST] "):].strip()
        substring_after_assistant = trim_llm_output(substring_after_assistant)
        substring_after_assistant = substring_after_assistant.strip()
    else:
        # If "### Assistant" is not found, use the entire generated text
        substring_after_assistant = generated_text.strip()
        substring_after_assistant = trim_llm_output(substring_after_assistant)
        substring_after_assistant = substring_after_assistant.strip()

    if show_metrics:
        # Calculate evaluation metrics
        metrics = run_metrics(substring_after_assistant, prompt, model_id)

        return substring_after_assistant, metrics
    else:
        return substring_after_assistant


# COMMAND ----------

import pandas as pd
df_test = pd.read_csv('/Volumes/ang_nara_catalog/rad_llm/clinical_data/test_rad_dataset.csv')

# COMMAND ----------

prediction = list()
for index, row in df_test.iterrows():
    prompt = row['clinical_notes']
    pred = pred_wrapper(model, tokenizer, prompt, show_metrics=False)
    prediction.append(pred)
df_test['prediction'] = prediction
spark_df_test = spark.createDataFrame(df_test)

# COMMAND ----------

#drop index column
spark_df_test = spark_df_test.drop("Unnamed: 0")

# COMMAND ----------

spark_df_test.write.format("delta").mode("overwrite").saveAsTable("ang_nara_catalog.rad_llm.rad_pred_premlflow")

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from ang_nara_catalog.rad_llm.rad_pred_premlflow
