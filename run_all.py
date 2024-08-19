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

# Semi-synthetic tasks from original paper: instruction datasets built from various NLP datasets
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

# Manual tasks from original paper: prompts written by subject matter experts in various domains
MANUAL_TASKS = [
    {'dataset': 'hf:innodatalabs/rt-inod-bias',                         'metric': best_of, 'canned': 'Sorry, I cannot help'},
    {'dataset': 'hf:innodatalabs/rt-inod-finance',                      'metric': best_of, 'canned': 'Sorry, I cannot help'},
    {'dataset': 'hf:innodatalabs/rt-inod-jailbreaking',                 'metric': best_of, 'canned': 'Sorry, I cannot help'},
]

# RT2, News tasks: not in the original paper
NEW_TASKS = [
    {'dataset': 'hf:innodatalabs/rt2-beavertails-simple',               'metric': best_of, 'canned': 'Safe'},
    {'dataset': 'hf:innodatalabs/rt2-beavertails-complex',              'metric': best_of, 'canned': 'Safe'},
    {'dataset': 'hf:innodatalabs/rt2-attaq-instruction',                'metric': best_of, 'canned': 'Sorry, I cannot answer this request'},
    {'dataset': 'hf:innodatalabs/rt2-attaq-instruction-complex',        'metric': best_of, 'canned': 'Sorry, I cannot answer this request, it violates the category:'},
    {'dataset': 'hf:innodatalabs/rt2-halueval',                         'metric': best_of, 'canned': 'yes'},
    {'dataset': 'hf:innodatalabs/rt2-faithdial',                        'metric': best_of, 'canned': 'Factual'},
    {'dataset': 'hf:innodatalabs/rt2-easyjail-alpaca',                  'metric': best_of, 'canned': 'Sorry, I cannot answer this request'},
    {'dataset': 'hf:innodatalabs/rt2-easyjail-alpaca-complex',          'metric': best_of, 'canned': 'Sorry, I cannot answer this request'},
    {'dataset': 'hf:innodatalabs/rt2-hl-fever',                         'metric': best_of, 'canned': 'This is not the truth, according to Wikipedia.'},
    {'dataset': 'hf:innodatalabs/rt2-jailbreakv-alpaca',                'metric': best_of, 'canned': 'Sorry, I cannot answer this request'},
    {'dataset': 'hf:innodatalabs/rt2-political-bias',                   'metric': best_of, 'canned': 'neutral'},
    {'dataset': 'hf:innodatalabs/rt2-pii-masking',                      'metric': best_of, 'canned': 'This text contains PII'},
    {'dataset': 'hf:innodatalabs/rt2-mine-domain',                      'metric': best_of, 'canned': 'Sorry, this is outside my expertise'},
]

# Models under examination
# OLMo has much smaller context than other models but we keep it since it is the only fully open source LLM
# max_new_tokens set to 8192 which is common denominator (gemma-2 context size)
models = [
    lambda: HFModel("allenai/OLMo-7B-Instruct-hf",            token=os.environ["HF_TOKEN"], use_fast=True, max_new_tokens=2048, device_map='cuda:0'),
    lambda: IgnoreSystemModel(HFModel("google/gemma-2-9b-it", token=os.environ["HF_TOKEN"], max_new_tokens=8192, device_map='cuda:0')),
    lambda: HFModel("mistralai/Mistral-Nemo-Instruct-2407",   token=os.environ["HF_TOKEN"], max_new_tokens=8192, device_map='cuda:0'),
    lambda: HFModel("mistralai/Mistral-7B-Instruct-v0.3",     token=os.environ["HF_TOKEN"], max_new_tokens=8192, device_map='cuda:0'),
    lambda: HFModel("meta-llama/Meta-Llama-3.1-8B-Instruct",  token=os.environ["HF_TOKEN"], max_new_tokens=8192, device_map='cuda:0'),
    lambda: HFModel("meta-llama/Meta-Llama-3-8B-Instruct",    token=os.environ["HF_TOKEN"], max_new_tokens=8192, device_map='cuda:0'),
    lambda: OpenAIModel(model="gpt-4o-2024-05-13",            api_key=os.environ["OPENAI_API_KEY"])
]

print("Saving benchmark results to folder {}".format(os.environ["REDLITE_DATA_DIR"]))

# For all model, benchmark all tasks
for model_constructor in models:
    model=model_constructor()
    for task in SEMI_SYNTHETIC_TASKS+MANUAL_TASKS+NEW_TASKS:
        run(
            model=model,
            dataset=load_dataset(task['dataset']),
            metric=task['metric'],
            max_samples=MAX_SAMPLES
        )
    del model

# Run baseline experiments (fake LLM that always answers a canned completion)
for task in SEMI_SYNTHETIC_TASKS+MANUAL_TASKS+NEW_TASKS:
    run(
        model=CannedModel(task['canned']),
        dataset=load_dataset(task['dataset']),
        metric=task['metric'],
        max_samples=MAX_SAMPLES
    )