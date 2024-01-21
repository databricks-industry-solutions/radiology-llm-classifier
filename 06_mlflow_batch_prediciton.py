# Databricks notebook source
!pip install mlflow --upgrade

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import mlflow
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

data = spark.read.csv('/Volumes/ang_nara_catalog/rad_llm/clinical_data/test_batch_data_10_rows.csv', header=True)

# COMMAND ----------

# You can update the catalog and schema name containing the model in Unity Catalog if needed
CATALOG_NAME = "ang_nara_catalog"
SCHEMA_NAME = "rad_llm"
MODEL_NAME = f"{CATALOG_NAME}.{SCHEMA_NAME}.radllama2_7b_test"

# COMMAND ----------

# Load pyfunc model from UC
model_uri = "models:/{model_name}@radllama".format(model_name=MODEL_NAME)
loaded_model = mlflow.pyfunc.load_model(model_uri)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Prediction using PySpark dataframe though Vectorized Pandas UDF (Faster than a PySpark UDF)

# COMMAND ----------

from pyspark.sql.functions import pandas_udf, PandasUDFType
import pandas as pd

@pandas_udf('string', PandasUDFType.SCALAR)
def model_predict(s: pd.Series) -> pd.Series:
    """
    Apply the model's predict function to each element of the input Pandas Series.

    Parameters:
    - s (pd.Series): Input Pandas Series containing data for predictions.

    Returns:
    - pd.Series: Series containing model predictions for each input element.
    """
    try:
        # Apply the model's predict function to each element of the input series
        # Note: loaded_model should be defined elsewhere in the code
        predictions = s.apply(lambda x: loaded_model.predict(pd.Series([x])))

        # Ensure the length of predictions is the same as the input series
        if len(predictions) != len(s):
            raise ValueError("Length of predictions does not match the input series")

        return predictions

    except Exception as e:
        # Log the exception for debugging purposes
        print(f"Error during prediction: {str(e)}")
        # Return a series with None values in case of an error
        return pd.Series([None] * len(s))

# Apply the Pandas UDF to the DataFrame
prediction_df = data.withColumn("prediction", model_predict(data['clinical_notes']))

# Display the result
display(prediction_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Prediction using Pandas dataframe though Python function

# COMMAND ----------

import pandas as pd
pd_data = pd.read_csv('/Volumes/ang_nara_catalog/rad_llm/clinical_data/test_batch_data_10_rows.csv')

# COMMAND ----------

import pandas as pd

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
        predictions = s.apply(lambda x: loaded_model.predict(pd.Series([x])))

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
prediction_series = apply_model_to_series(clinical_notes_series, loaded_model)

# Combine the original data with predictions
result_df = pd.concat([pd_data, prediction_series.rename("prediction")], axis=1)

# Display the result
display(result_df)

