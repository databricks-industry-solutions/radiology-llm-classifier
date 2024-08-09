# Databricks notebook source
# MAGIC %md # Fine Tuning an LLM

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC 01_train_llm - Load sample data, configure fine tuning parameters, load model from hugging face, fine tune the model based upon sample data & save model weights

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC 02_merge_model_weights - Combine fine tuned weights with model and save finalized model to Databricks or Hugging Face

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC 03_mlflow_batch_prediction - Serve model via MLFlow and run batch predictions on sample data

# COMMAND ----------

# MAGIC %md 
# MAGIC 04_model_evaluation - Compare fine tuned model results to actual radiology labels
