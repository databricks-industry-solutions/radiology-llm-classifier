![image](https://raw.githubusercontent.com/databricks-industry-solutions/.github/main/profile/solacc_logo_wide.png)

[![CLOUD](https://img.shields.io/badge/CLOUD-ALL-blue?logo=googlecloud&style=for-the-badge)](https://cloud.google.com/databricks)
[![POC](https://img.shields.io/badge/POC-10_days-green?style=for-the-badge)](https://databricks.com/try-databricks)

# GenAI for Radiology Protocol Labeling

## Use Case 

Radiology examination protocols and labeling are a time consuming administrative process for clinicians today. This repo explores fine tining an LLM on Databricks to automate label generation from clinical notes. More details around this use case can be found [here](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8861685/)

## Running the Solution

This solution is designed to run notebooks 01_train_llm... through 04_model_evaluation in numerical order. 

### GPUs for Compute 

Recommend using NVIDIA A10G or NVIDIA T4 GPU which are, according to AWS, ‘optimized for machine learning inference and small scale training’. Fine tuning exercise completed on AWS with 2-8 autoscaling workers of type g5.24xlarge GPUs.

### Sample Dataset

Sample datasets in data/ are generated synthetic data from ChatGPT and are for the sole purpose of demonstrating how this pipeline for fine tuning would run in an environment.

### Fine Tuning the Model

We use a base [model](https://huggingface.co/epfl-llm/meditron-7b) from Hugging Face, [Meditron](https://github.com/epfLLM/meditron), to fine tune from the [data](data/) in this repo. This model is chosen due to its quality in Healthcare having been trained on pubmed data as well as its open source license under Apache2. This model can be easily subbed out for a different model in the notebook AARON TODO

#### Optimizing tuning Parameters 

![image](https://github.com/databricks-industry-solutions/radiology-llm-classifier/blob/feature-new-model-metrics/images/weights.png?raw=true)

LoRA presents an enhanced fine-tuning technique that diverges from the conventional approach of full fine-tuning all weights within the pre-trained large language model's weight matrix. Instead, LoRA fine-tunes two smaller matrices, serving as approximations of the larger matrix. These smaller matrices constitute what is known as the LoRA adapter. Once fine-tuned, this adapter is incorporated into the pre-trained model for inference purposes.

QLoRA represents a refinement of LoRA, prioritizing memory efficiency. It achieves this by further optimizing the weights of the LoRA adapters through quantization, reducing their precision from 8-bit to 4-bit. This reduction significantly diminishes the memory footprint and storage demands. In the QLoRA framework, the pre-trained model is loaded into GPU memory with these quantized 4-bit weights, as opposed to the 8-bit precision utilized in LoRA.

Despite the decrease in bit precision, QLoRA maintains a comparable level of effectiveness to its predecessor, LoRA. 

Through the integration of QLoRA in this repository, we managed to reduce the training duration to approximately 2 hours on a g5.24xlarge driver and worker setup for 30k training samples. Without QLoRA, we anticipate a significantly longer training time, necessitating larger memory-sized driver and worker nodes. By employing QLoRA, we not only minimize the number of GPUs needed but also decrease the memory footprint of the model weights.

### Notes on Accuracy 

To assess the precision of our fine-tuned model, we utilized Semantic Similarity Search, which gauges the alignment between the ground truth and predictions on a 0-1 scale within our 200 test batch test dataset. A score of 0 denotes no similarity, while 1 signifies exact correspondence. Our evaluation yielded an impressive average score of 0.86 on the batch test dataset.

![image](https://github.com/databricks-industry-solutions/radiology-llm-classifier/blob/feature-new-model-metrics/images/clinical_notes.png?raw=true)

## Project support 

Please note the code in this project is provided for your exploration only, and are not formally supported by Databricks with Service Level Agreements (SLAs). They are provided AS-IS and we do not make any guarantees of any kind. Please do not submit a support ticket relating to any issues arising from the use of these projects. The source in this project is provided subject to the Databricks [License](./LICENSE). All included or referenced third party libraries are subject to the licenses set forth below.

Any issues discovered through the use of this project should be filed as GitHub Issues on the Repo. They will be reviewed as time permits, but there are no formal SLAs for support. 
