# Databricks notebook source
client.create_endpoint(
    name="bedrock-anthropic-completions-endpoint",
    config={
        "served_entities": [
            "external_model": {
                "name": "claude-v2",
                "provider": "aws-bedrock",
                "task": "llm/v1/completions",
                "aws_bedrock_config": {
                    "aws_region": GLD_AWS_REGION,
                    "aws_access_key_id": GLD_AWS_ACCESS_KEY_ID,
                    "aws_secret_access_key": GLD_AWS_SECRET_ACCESS_KEY,
                    "bedrock_provider": "anthropic"
                }
            }
        ]
    }
)

# COMMAND ----------

pip install mlflow --upgrade

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

GLD_AWS_REGION = 'x'
GLD_AWS_ACCESS_KEY_ID = 'y'
GLD_AWS_SECRET_ACCESS_KEY = 'z'

# COMMAND ----------

import mlflow.deployments

client = mlflow.deployments.get_deploy_client("databricks")

client.create_endpoint(
    name="bedrock-anthropic-completions-endpoint",
    config={
        "served_entities": [
            {
                "external_model": {
                    "name": "claude-v2",
                    "provider": "aws-bedrock",
                    "task": "llm/v1/completions",
                    "aws_bedrock_config": {
                        "aws_region": GLD_AWS_REGION,
                        "aws_access_key_id": GLD_AWS_ACCESS_KEY_ID,
                        "aws_secret_access_key": GLD_AWS_SECRET_ACCESS_KEY,
                        "bedrock_provider": "anthropic"
                    }
                }
            }
        ]
    }
)

# COMMAND ----------

token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
dbutils.fs.put("file:///root/.databrickscfg", "[DEFAULT]\nhost=https://e2-demo-field-eng.cloud.databricks.com\ntoken=" + token, overwrite=True)
