# Databricks notebook source
dbutils.widgets.text("model_version", "1")
dbutils.widgets.text("pred_table", "ang_nara_catalog.rad_llm.batch_pred_res")

# COMMAND ----------

#install libraries
!pip install -q accelerate==0.21.0 peft==0.4.0 bitsandbytes==0.40.2 transformers==4.31.0 trl==0.4.7 guardrail-ml==0.0.12
!pip install -q unstructured["local-inference"]==0.7.4 pillow
!pip install pydantic==1.8.2 
dbutils.library.restartPython()

# COMMAND ----------

import os
import torch
import pandas as pd
import re
import mlflow
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
from mlflow.models.signature import infer_signature
from mlflow.pyfunc import PythonModel
from mlflow.models.signature import infer_signature
from mlflow.transformers import generate_signature_output

# COMMAND ----------

class LLMModelWrapper(PythonModel):
    """
    Wrapper class for LLM with MLflow integration.

    Parameters:
    - model: Fine-tuned language model for text generation.
    - tokenizer: Tokenizer corresponding to the language model.
    - snapshot_location: Path to save the model artifacts.

    Attributes:
    - model: Fine-tuned language model for text generation.
    - tokenizer: Tokenizer corresponding to the language model.
    - snapshot_location: Path to save the model artifacts.
    """

    def __init__(self, model, tokenizer, snapshot_location):
        self.model = model
        self.tokenizer = tokenizer
        self.snapshot_location = snapshot_location

    def load_context(self, context):
        """
        Load model-specific artifacts during model loading.

        Parameters:
        - context: MLflow model context containing model-specific artifacts.
        """
        # Load the fine-tuned LLM model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(context.artifacts["model_path"])
        self.tokenizer = AutoTokenizer.from_pretrained(context.artifacts["tokenizer_path"])

    @staticmethod
    def trim_llm_output(text):
        """
        Trim the generated text to remove any trailing punctuation.

        Parameters:
        - text: Generated text to be trimmed.

        Returns:
        - Trimmed text with trailing punctuation removed.
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

    def predict(self, context, model_input):
        """
        Generate text based on the given input using the pre-trained language model.

        Parameters:
        - context: MLflow model context (not used in this implementation).
        - model_input: Input prompt for text generation.

        Returns:
        - Generated text based on the input prompt.
        """
        prompt = model_input
        temp = 0.1
        max_length = 100
        show_metrics = False

        pipe = pipeline(task="text-generation",
                        model=self.model,
                        tokenizer=self.tokenizer,
                        max_length=max_length,
                        do_sample=True,
                        temperature=temp)

        result = pipe(f"<s>[INST] {prompt} [/INST]")
        generated_text = result[0]['generated_text']

        # Find the index of "[/INST]" in the generated text
        index = generated_text.find("[/INST] ")
        if index != -1:
            # Extract the substring after "[/INST]"
            substring_after_inst = generated_text[index + len("[/INST] "):].strip()
            substring_after_inst = self.trim_llm_output(substring_after_inst)
            substring_after_inst = substring_after_inst.strip()
        else:
            # If "[/INST]" is not found, use the entire generated text
            substring_after_inst = generated_text.strip()
            substring_after_inst = self.trim_llm_output(substring_after_inst)
            substring_after_inst = substring_after_inst.strip()

        if show_metrics:
            # Calculate evaluation metrics
            metrics = self.run_metrics(substring_after_inst, prompt)
            return substring_after_inst, metrics
        else:
            return substring_after_inst

    def predict_wrapper(self, model_input):
        """
        Wrapper function for predict method without the context argument.

        Parameters:
        - model_input: Input prompt for text generation.

        Returns:
        - Generated text based on the input prompt.
        """
        result = self.predict(None, model_input)
        return result

    def save(self):
        """
        Save the model and tokenizer as artifacts.
        """
        self.model.save_pretrained(self.snapshot_location)
        self.tokenizer.save_pretrained(self.snapshot_location)

# COMMAND ----------

loaded_model = AutoModelForCausalLM.from_pretrained("/Volumes/ang_nara_catalog/rad_llm/results/model")
loaded_tokenizer = AutoTokenizer.from_pretrained("/Volumes/ang_nara_catalog/rad_llm/results/model")

# COMMAND ----------

# Assuming you have a trained LLM model
llm_model = loaded_model.from_pretrained("/Volumes/ang_nara_catalog/rad_llm/results/model")
llm_tokenizer = loaded_tokenizer.from_pretrained("/Volumes/ang_nara_catalog/rad_llm/results/model")
snapshot_location = os.path.expanduser("~/.cache/huggingface/model")

# Create an instance of LLMModelWrapper
llm_model_wrapper = LLMModelWrapper(llm_model, llm_tokenizer, snapshot_location)

# Save the llm model and tokenizer as artifacts
llm_model_wrapper.save()

input_example = 'Notes: rule out renal recurrence  History: Renal cell carcinoma, sp partial nephrectomy'

# Log the model with MLflow
with mlflow.start_run():
    mlflow.pyfunc.log_model(
        "rad-meditron7b",
        python_model=llm_model_wrapper,
        artifacts={'model_path': snapshot_location, 'tokenizer_path': snapshot_location},
        extra_pip_requirements=[
            "accelerate==0.21.0",
            "tensorflow==2.13.0",
            "transformers==4.31.0",
            "deepspeed==0.10.0",
            "torch==2.0.1"
        ],
        input_example=input_example,
        signature=infer_signature(
            model_input=input_example,
            model_output="CT Dedicated Kidney"
        )
    )
    run_id = mlflow.active_run().info.run_id
    catalog = "ang_nara_catalog"
    schema = "rad_llm"
    model_name = "rad-meditron7b"
    mlflow.set_registry_uri("databricks-uc")
    mlflow.register_model(
        model_uri=f"runs:/{run_id}/rad-meditron7b",
        name=f"{catalog}.{schema}.{model_name}"
    )

# COMMAND ----------

# You can update the catalog and schema name containing the model in Unity Catalog if needed
CATALOG_NAME = "ang_nara_catalog"
SCHEMA_NAME = "rad_llm"
MODEL_NAME = f"{CATALOG_NAME}.{SCHEMA_NAME}.rad-meditron"

# COMMAND ----------

model_version = dbutils.widgets.get("model_version")

# COMMAND ----------

# Load pyfunc model from UC
import mlflow
mlflow.set_registry_uri("databricks-uc")
model_uri = "models:/{model_name}/{model_version}".format(model_name=MODEL_NAME, model_version=model_version)
model =  mlflow.pyfunc.load_model(model_uri)

# COMMAND ----------

pd_data = pd.read_csv('/Volumes/ang_nara_catalog/rad_llm/clinical_data/batch_data_test.csv')
pd_data = pd_data.drop(['Unnamed: 0'], axis=1)

# COMMAND ----------

def apply_model_to_series(s, loaded_model):
    """
    Apply LLM to a Pandas Series.

    Parameters:
    - s (pd.Series): The Pandas Series to which the model will be applied.
    - loaded_model: The fine-tuned LLM model.

    Returns:
    - pd.Series: Predictions from the model for each element in the input series.
    """
    try:
        # Apply the model's predict function to each element of the input series
        predictions = s.apply(lambda x: model.predict(pd.Series([x])))

        # Ensure the length of predictions is the same as the input series
        if len(predictions) != len(s):
            raise ValueError("Length of predictions does not match the input series")

        return predictions

    except Exception as e:
        # Log the exception for debugging purposes
        print(f"Error during prediction: {str(e)}")
        return pd.Series([None] * len(s))

# Assuming pd_data is your Pandas DataFrame
clinical_notes_series = pd_data['clinical_notes']

# Apply the function to get prediction_series
prediction_series = apply_model_to_series(clinical_notes_series, model)

# Combine the original data with predictions
result_df = pd.concat([pd_data, prediction_series.rename("predictions")], axis=1)

# Display the result
display(result_df)

# COMMAND ----------

spark_batch_df = spark.createDataFrame(result_df)
spark_batch_df.write.format("delta").mode("overwrite").option("mergeSchema", "true").saveAsTable(pred_table)
