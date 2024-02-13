# Databricks notebook source
#install libraries
!pip install -q accelerate==0.21.0 peft==0.4.0 bitsandbytes==0.40.2 transformers==4.31.0 trl==0.4.7 guardrail-ml==0.0.12
!pip install -q unstructured["local-inference"]==0.7.4 pillow
!pip install pydantic==1.8.2 

# COMMAND ----------

dbutils.library.restartPython()

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
from guardrail.client import run_metrics

# COMMAND ----------

val = dbutils.secrets.get(scope="ang_token", key="token")

# COMMAND ----------

!huggingface-cli login --token $val

# COMMAND ----------

import os
from transformers import AutoTokenizer, AutoModelForCausalLM
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

import pandas as pd
df_test = pd.read_csv('/Volumes/ang_nara_catalog/rad_llm/clinical_data/test_rad_dataset.csv')

# COMMAND ----------

def apply_pred_wrapper_to_df(df, model, tokenizer, show_metics=False):
    """
    Apply the pred_wrapper function to each row of a DataFrame.

    Parameters:
    - df (pd.DataFrame): Input DataFrame containing 'clinical_notes' column.
    - model: The LLM model.
    - tokenizer: The tokenizer used for processing the input.
    - show_metrics (bool): Whether to show metrics during prediction.

    Returns:
    - pd.DataFrame: DataFrame with an additional 'prediction' column.
    """
    # Create an empty list to store predictions
    prediction = []

    # Iterate over each row of the DataFrame
    for index, row in df.iterrows():
        # Extract the 'clinical_notes' value from the current row
        prompt = row['clinical_notes']

        # Apply the pred_wrapper function to obtain predictions
        pred = pred_wrapper(model, tokenizer, prompt, show_metrics=False)

        # Append the prediction to the list
        prediction.append(pred)

    # Add the 'prediction' column to the original DataFrame
    df['prediction'] = prediction

    return df

# Example usage:
result_df = apply_pred_wrapper_to_df(df_test, model, tokenizer)

# Display the result
display(result_df)

