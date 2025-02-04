#################################################################### 
#                                                                  #
# Benchmarking Llama2, Mistral, Gemma and GPT                      #
# for Factuality, Toxicity, Bias and Propensity for Hallucinations #
#                                                                  #
####################################################################
import os
from redlite import run, load_dataset
from redlite.model.hf_model import HFModel
from redlite.model import IgnoreSystemModel
from redlite.metric.rouge import RougeMetric

# Evaluation metric
reply_with_one_word = RougeMetric(rouge_type='rouge1')

# We run benchmark on the first 250 prompts for each dataset
MAX_SAMPLES = 250

# Toxicity task made of two custom datasets
TOXICITY_TASKS = [
    {'dataset': 'hf:innodatalabs/rtb-toxigen-data',                             'metric': reply_with_one_word, 'canned': 'Acceptable Unacceptable'},
    {'dataset': 'hf:innodatalabs/rtb-real-toxicity',                            'metric': reply_with_one_word, 'canned': 'Acceptable Unacceptable'},
]

# Models under examination
# Mistral-Nemo was the best from August benchmark
# OLMo is in the evalutaion because it is fully open source LLM
# Llama-Guard is the safe version of Llama
# Llama-instruct is the base model
# Llama-instruct-Uncensored is an uncensored version of Llama
# Mistral-grok mimics xAI Grok
models = [
    
    lambda: HFModel("mistralai/Mistral-Nemo-Instruct-2407",                                          token=os.environ["HF_TOKEN"], max_new_tokens=8192, device_map='cuda:0'),
    lambda: HFModel("allenai/OLMo-7B-0724-Instruct-hf",                                              token=os.environ["HF_TOKEN"], use_fast = True, max_new_tokens=2048, device_map='cuda:0'),
    lambda: IgnoreSystemModel(HFModel("meta-llama/Llama-Guard-3-8B",                                 token=os.environ["HF_TOKEN"], max_new_tokens=8192, device_map='cuda:0')),
    lambda: HFModel("meta-llama/Llama-3.1-8B-Instruct",                                              token=os.environ["HF_TOKEN"], max_new_tokens=8192, device_map='cuda:0'),
    lambda: HFModel("Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2",                                    token=os.environ["HF_TOKEN"], max_new_tokens=8192, device_map='cuda:0'),
    lambda: HFModel("HuggingFaceH4/mistral-7b-grok",                                                 token=os.environ["HF_TOKEN"], max_new_tokens=8192, device_map='cuda:0'),
    
]

print("Saving benchmark results to folder {}".format(os.environ["REDLITE_DATA_DIR"]))

# For all model, benchmark all tasks
for model_constructor in models:
    model=model_constructor()
    for task in TOXICITY_TASKS:
        run(
            model=model,
            dataset=load_dataset(task['dataset']),
            metric=task['metric'],
            max_samples=MAX_SAMPLES
        )
    del model