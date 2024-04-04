#################################################################### 
#                                                                  #
# Benchmarking Llama2, Mistral, Gemma and GPT                      #
# for Factuality, Toxicity, Bias and Propensity for Hallucinations #
#                                                                  #
####################################################################
import os
from redlite import run, load_dataset
from redlite.model.hf_model import HFModel
from redlite.model.openai_model import OpenAIModel
from redlite.model import IgnoreSystemModel, CannedModel
from redlite.metric import MatchMetric, BestOfMetric
from redlite.metric.rouge import RougeMetric

# Evaluation metric
best_of = BestOfMetric(
    RougeMetric(rouge_type='rouge2'),
    MatchMetric(ignore_case=True, ignore_punct=True, strategy='prefix'),
)

# We run benchmark on the first 250 prompts for each dataset
MAX_SAMPLES = 250

# Semi-synthetic tasks: instruction datasets built from various NLP datasets
SEMI_SYNTHETIC_TASKS = [
    {'dataset': 'hf:innodatalabs/rt-cogensumm',                         'metric': best_of, 'canned': 'N/A'},
    {'dataset': 'hf:innodatalabs/rt-factcc',                            'metric': best_of, 'canned': 'CORRECT'},
    {'dataset': 'hf:innodatalabs/rt-frank',                             'metric': best_of, 'canned': 'Good'},
    {'dataset': 'hf:innodatalabs/rt-gsm8k-gaia',                        'metric': best_of, 'canned': 'Can solve'},
    {'dataset': 'hf:innodatalabs/rt-realtoxicity-grammatical',          'metric': best_of, 'canned': 'I cannot correct this sentence'},
    {'dataset': 'hf:innodatalabs/rt-realtoxicity-grammatical-multiturn','metric': best_of, 'canned': 'I cannot correct this sentence'},
    {'dataset': 'hf:innodatalabs/rt-realtoxicity-paraphrasing',         'metric': best_of, 'canned': 'I cannot paraphrase this sentence'},
    {'dataset': 'hf:innodatalabs/rt-realtoxicity-poetry',               'metric': best_of, 'canned': 'I cannot write a poem from this sentence'},
    {'dataset': 'hf:innodatalabs/rt-realtoxicity-translation',          'metric': best_of, 'canned': 'I cannot translate this sentence'},
    {'dataset': 'hf:innodatalabs/rt-realtoxicity-translation-multiturn','metric': best_of, 'canned': 'I cannot translate this sentence'},
    {'dataset': 'hf:innodatalabs/rt-realtoxicity-verlan',               'metric': best_of, 'canned': 'I cannot decrypt this sentence'},
]

# Manual tasks: prompts written by subject matter experts in various domains
MANUAL_TASKS = [
    {'dataset': 'hf:innodatalabs/rt-inod-bias',                         'metric': best_of, 'canned': 'Sorry, I cannot help'},
    {'dataset': 'hf:innodatalabs/rt-inod-finance',                      'metric': best_of, 'canned': 'Sorry, I cannot help'},
    {'dataset': 'hf:innodatalabs/rt-inod-jailbreaking',                 'metric': best_of, 'canned': 'Sorry, I cannot help'},
]

# Models under examination
models = [
    lambda: HFModel("meta-llama/Llama-2-13b-chat-hf", token=os.environ["HF_TOKEN"], max_length=4096, device_map='cuda:0'),
    lambda: HFModel("meta-llama/Llama-2-7b-chat-hf", token=os.environ["HF_TOKEN"], max_length=4096, device_map='cuda:0'),
    lambda: IgnoreSystemModel(HFModel("mistralai/Mistral-7B-Instruct-v0.1", token=os.environ["HF_TOKEN"], device_map='cuda:0')),
    lambda: IgnoreSystemModel(HFModel("mistralai/Mistral-7B-Instruct-v0.2", token=os.environ["HF_TOKEN"], device_map='cuda:0')),
    lambda: IgnoreSystemModel(HFModel("google/gemma-2b-it", token=os.environ["HF_TOKEN"], device_map='cuda:0')),
    lambda: IgnoreSystemModel(HFModel("google/gemma-7b-it", token=os.environ["HF_TOKEN"], device_map='cuda:0')),   
    lambda: OpenAIModel(model="gpt-3.5-turbo-1106", api_key=os.environ["OPENAI_API_KEY"]),
    lambda: OpenAIModel(model="gpt-4-0125-preview", api_key=os.environ["OPENAI_API_KEY"])
]

print("Saving benchmark results to folder {}".format(os.environ["REDLITE_DATA_DIR"]))

# For all model, benchmark all tasks
for model_constructor in models:
    model=model_constructor()
    for task in SEMI_SYNTHETIC_TASKS+MANUAL_TASKS:
        run(
            model=model,
            dataset=load_dataset(task['dataset']),
            metric=task['metric'],
            max_samples=MAX_SAMPLES
        )
    del model

# Run baseline experiments (fake LLM that always answers a canned completion)
for task in SEMI_SYNTHETIC_TASKS+MANUAL_TASKS:
    run(
        model=CannedModel(task['canned']),
        dataset=load_dataset(task['dataset']),
        metric=task['metric'],
        max_samples=MAX_SAMPLES
    )