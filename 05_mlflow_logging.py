# Databricks notebook source
pip install mlflow --upgrade

# COMMAND ----------

#install libraries
!pip install -q accelerate==0.21.0 peft==0.4.0 bitsandbytes==0.40.2 transformers==4.31.0 trl==0.4.7 guardrail-ml==0.0.12
!pip install -q unstructured["local-inference"]==0.7.4 pillow

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import mlflow
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
from guardrail.client import run_metrics

# COMMAND ----------

!huggingface-cli login --token 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

# COMMAND ----------

tokenizer = AutoTokenizer.from_pretrained("RadiologyLLMs/RadLlama2-7b")
model = AutoModelForCausalLM.from_pretrained("RadiologyLLMs/RadLlama2-7b")

# COMMAND ----------

import re
def trim_llm_output(text):
    """
    Trims the given text at the first occurrence of punctuation.

    Parameters:
    - text (str): The input text to be trimmed.

    Returns:
    - str: The trimmed text up to the first occurrence of punctuation, or the original text if no punctuation is found.
    """

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
def pred_wrapper(model, tokenizer, prompt, model_id=1, temp=0.1, max_length=1, show_metrics=False):
    """
    Wrapper function for text generation using a transformer model.

    Args:
        model (str): The transformer model to use.
        tokenizer (str): The tokenizer for the specified model.
        prompt (str): The input prompt for text generation.
        model_id (int, optional): Identifier for the model. Defaults to 1.
        show_metrics (bool, optional): Whether to calculate and display evaluation metrics. Defaults to True.
        temp (float, optional): Temperature parameter for sampling. Defaults to 0.1.
        max_length (int, optional): Maximum length of generated text. Defaults to 1.

    Returns:
        tuple or str: If show_metrics is True, returns a tuple containing the generated text and evaluation metrics.
                      If show_metrics is False, returns only the generated text.
    """
    # Create a text generation pipeline using the specified model and tokenizer
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

def predict (prompt):
  result = pred_wrapper(model, tokenizer, prompt, show_metrics=False)
  return result

# COMMAND ----------

input_example = 'Notes: rule out renal recurrence  History: Renal cell carcinoma, sp partial nephrectomy'

# COMMAND ----------

#define input and output format of model
from mlflow.models.signature import infer_signature
from mlflow.transformers import generate_signature_output
signature = infer_signature(
  model_input=input_example,
  model_output="CT Dedicated Kidney"
)

# COMMAND ----------

import tempfile
import os

temp_dir = tempfile.TemporaryDirectory()
tokenizer_path = os.path.join(temp_dir.name, "tokenizer")

tokenizer.save_pretrained(tokenizer_path)

model_path = os.path.join(temp_dir.name, "model")
model.save_pretrained(model_path)

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment("/Users/narasimha.kamathardi@databricks.com/Project: radiology label prediction using LLMs/code/sandbox")

with mlflow.start_run():
    mlflow.pyfunc.log_model(
        "ang_nara_catalog.rad_llm.radllama2_7b_test",
        python_model=predict,
        artifacts={"tokenizer": tokenizer_path, "model": model_path},
        input_example=input_example,
        signature=signature
    )
    run_id = mlflow.active_run().info.run_id
    catalog = "ang_nara_catalog"
    schema = "rad_llm"
    model_name = "radllama2_7b_test"
    mlflow.set_registry_uri("databricks-uc")
    mlflow.register_model(
        model_uri="runs:/"+run_id+"/ang_nara_catalog.rad_llm.radllama2_7b_test",
        name=f"{catalog}.{schema}.{model_name}")
    
temp_dir.cleanup()
