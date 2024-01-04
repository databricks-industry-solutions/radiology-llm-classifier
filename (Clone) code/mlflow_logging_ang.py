# Databricks notebook source
import mlflow
from transformers import AutoTokenizer, AutoModelForCausalLM

# COMMAND ----------

!huggingface-cli login --token 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

# COMMAND ----------

tokenizer = AutoTokenizer.from_pretrained("mewtoo/RadLlama2")
model = AutoModelForCausalLM.from_pretrained("mewtoo/RadLlama2")

# COMMAND ----------

import re
def trim_llm_output(text):
    # Define a regular expression to match any punctuation
    punctuation_regex = re.compile(r'[.,;!?://()]+')

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

def pred_wrapper(model, tokenizer, prompt, model_id=1, show_metrics=True, temp=0.1, max_length=1):
    # Suppress Hugging Face pipeline logging
    logging.set_verbosity(logging.CRITICAL)

    # Initialize the pipeline
    pipe = pipeline(task="text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_length=max_length,
                    do_sample=True,
                    temperature=temp)

    # Generate text using the pipeline
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

input_example = 'Reason: 55M motor vehicle accident, lower back pain. History: Trauma-related pain with limited mobility.'

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
mlflow.set_experiment("/Users/narasimha.kamathardi@databricks.com/Project: radiology label prediction using LLMs/code/05_mlflow_logging")

with mlflow.start_run():
    mlflow.pyfunc.log_model(
        "ang_nara_catalog.rad_llm.radllama2_7b",
        python_model=predict,
        artifacts={"tokenizer": tokenizer_path, "model": model_path},
        input_example=input_example,
        signature=signature
    )
    run_id = mlflow.active_run().info.run_id
    catalog = "ang_nara_catalog"
    schema = "rad_llm"
    model_name = "radllama2_7b"
    mlflow.set_registry_uri("databricks-uc")
    mlflow.register_model(
        model_uri="runs:/"+run_id+"/ang_nara_catalog.rad_llm.radllama2_7b",
        name=f"{catalog}.{schema}.{model_name}")
    
temp_dir.cleanup()
