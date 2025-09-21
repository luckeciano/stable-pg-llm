# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Custom evaluation tasks for LightEval."""

import random

from open_r1.utils.custom_metrics import make_metrics_at_k, multilingual_extractive_match_metric_robust

from lighteval.metrics.dynamic_metrics import (
    ExprExtractionConfig,
    IndicesExtractionConfig,
    LatexExtractionConfig,
    multilingual_extractive_match_metric,
)
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.utils.language import Language


# Prompt template adapted from
# - simple-evals: https://github.com/openai/simple-evals/blob/6e84f4e2aed6b60f6a0c7b8f06bbbf4bfde72e58/math_eval.py#L17
# - Llama 3: https://huggingface.co/datasets/meta-llama/Llama-3.2-1B-Instruct-evals/viewer/Llama-3.2-1B-Instruct-evals__math__details?views%5B%5D=llama_32_1b_instruct_evals__math__details
# Note that it is important to have the final answer in a box for math-verify to work correctly
MATH_QUERY_TEMPLATE = """
Solve the following math problem efficiently and clearly.  The last line of your response should be of the following format: 'Therefore, the final answer is: $\\boxed{{ANSWER}}$. I hope it is correct' (without quotes) where ANSWER is just the final number or expression that solves the problem. Think step by step before answering.

{Question}
""".strip()

# Prompt template from simple-evals: https://github.com/openai/simple-evals/blob/83ed7640a7d9cd26849bcb3340125002ef14abbe/common.py#L14
GPQA_QUERY_TEMPLATE = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{Question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()

# k_values = [1, 4, 8, 16, 32, 64, 96, 128]
k_values = [1]

print(f"k_values: {k_values}")

latex_gold_metric = multilingual_extractive_match_metric_robust(
    language=Language.ENGLISH,
    fallback_mode="first_match",
    precision=5,
    gold_extraction_target=(LatexExtractionConfig(),),
    # Match boxed first before trying other regexes
    pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig(boxed_match_priority=0)),
    aggregation_function=max,
    timeout_seconds=1
)

expr_gold_metric = multilingual_extractive_match_metric_robust(
    language=Language.ENGLISH,
    fallback_mode="first_match",
    precision=5,
    gold_extraction_target=(ExprExtractionConfig(),),
    # Match boxed first before trying other regexes
    pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig(boxed_match_priority=0)),
    aggregation_function=max,
    timeout_seconds=1
)

gpqa_metric = multilingual_extractive_match_metric_robust(
    language=Language.ENGLISH,
    gold_extraction_target=[IndicesExtractionConfig(prefix_for_extraction="NativeLetters")],
    pred_extraction_target=[IndicesExtractionConfig(prefix_for_extraction="NativeLetters")],
    precision=5,
    timeout_seconds=1
)

minerva_metric = multilingual_extractive_match_metric_robust(
    language=Language.ENGLISH,
    fallback_mode="first_match",
    precision=5,
    gold_extraction_target=(LatexExtractionConfig(), ExprExtractionConfig()),
    # Match boxed first before trying other regexes
    pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig(boxed_match_priority=0)),
    aggregation_function=max,
    timeout_seconds=1
)

olympiadbench_metric = multilingual_extractive_match_metric_robust(
    language=Language.ENGLISH,
    precision=5,
    gold_extraction_target=(LatexExtractionConfig(), ExprExtractionConfig()),
    pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig(boxed_match_priority=0)),
    timeout_seconds=1
)

def math_prompt_fn(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=MATH_QUERY_TEMPLATE.format(Question=line["problem"]),
        choices=[line["solution"]],
        gold_index=0,
    )


def aime_prompt_fn(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=MATH_QUERY_TEMPLATE.format(Question=line["problem"]),
        choices=[line["answer"]],
        gold_index=0,
    )


def gpqa_prompt_fn(line, task_name: str = None):
    gold_index = random.randint(0, 3)
    choices = [line["Incorrect Answer 1"], line["Incorrect Answer 2"], line["Incorrect Answer 3"]]
    choices.insert(gold_index, line["Correct Answer"])
    query = GPQA_QUERY_TEMPLATE.format(
        A=choices[0], B=choices[1], C=choices[2], D=choices[3], Question=line["Question"]
    )
    return Doc(
        task_name=task_name,
        query=query,
        choices=["A", "B", "C", "D"],
        gold_index=gold_index,
        instruction=query,
    )

def minervamath_prompt_fn(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=MATH_QUERY_TEMPLATE.format(Question=line["question"]),
        choices=[f"The solution is \\boxed{{{line['answer']}}}"],
        gold_index=0,
    )

def gsm8k_prompt_fn(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=MATH_QUERY_TEMPLATE.format(Question=line["problem"]),
        choices=[line["expected_answer"]],
        gold_index=0,
    )

def amc23_prompt_fn(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=MATH_QUERY_TEMPLATE.format(Question=line["question"]),
        choices=[str(line["answer"])],
        gold_index=0,
    )

def olympiadbench_prompt_fn(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=MATH_QUERY_TEMPLATE.format(Question=line["question"]),
        choices=[f"The solution is \\boxed{{{line['final_answer']}}}"],
        gold_index=0,
    )

# Define tasks
aime24 = LightevalTaskConfig(
    name="aime24",
    suite=["custom"],
    prompt_function=aime_prompt_fn,
    hf_repo="HuggingFaceH4/aime_2024",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],    
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metric=[expr_gold_metric] + make_metrics_at_k(k_values, expr_gold_metric),
    version=1,
    num_samples=max(k_values)
)
aime25 = LightevalTaskConfig(
    name="aime25",
    suite=["custom"],
    prompt_function=aime_prompt_fn,
    hf_repo="yentinglin/aime_2025",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metric=[expr_gold_metric] + make_metrics_at_k(k_values, expr_gold_metric),
    version=1,
    num_samples=max(k_values)
)
math_500 = LightevalTaskConfig(
    name="math_500",
    suite=["custom"],
    prompt_function=math_prompt_fn,
    hf_repo="HuggingFaceH4/MATH-500",
    hf_subset="default",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metric=[latex_gold_metric] + make_metrics_at_k(k_values, latex_gold_metric),
    version=1,
    num_samples=max(k_values)
)
gpqa_diamond = LightevalTaskConfig(
    name="gpqa:diamond",
    suite=["custom"],
    prompt_function=gpqa_prompt_fn,
    hf_repo="Idavidrein/gpqa",
    hf_subset="gpqa_diamond",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,  # needed for reasoning models like R1
    metric=[gpqa_metric] + make_metrics_at_k(k_values, gpqa_metric),
    stop_sequence=[],  # no stop sequence, will use eos token
    trust_dataset=True,
    version=1,
    num_samples=max(k_values)
)

minervamath = LightevalTaskConfig( 
    name="minervamath",
    suite=["custom"],
    prompt_function=minervamath_prompt_fn,
    hf_repo="math-ai/minervamath",
    hf_subset="default",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metric=[minerva_metric] + make_metrics_at_k(k_values, minerva_metric),
    version=1, 
    num_samples=max(k_values)
)

gsm8k = LightevalTaskConfig( 
    name="gsm8k",
    suite=["custom"],
    prompt_function=gsm8k_prompt_fn,
    hf_repo="zwhe99/gsm8k",
    hf_subset="default",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metric=[expr_gold_metric] + make_metrics_at_k(k_values, expr_gold_metric),
    version=1, 
    num_samples=max(k_values)
)

amc23 = LightevalTaskConfig( 
    name="amc23",
    suite=["custom"],
    prompt_function=amc23_prompt_fn,
    hf_repo="zwhe99/amc23",
    hf_subset="default",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metric=[expr_gold_metric] + make_metrics_at_k(k_values, expr_gold_metric),
    version=1, 
    num_samples=max(k_values)
)

olympiadbench = LightevalTaskConfig(
    name="olympiadbench",
    suite=["custom"],
    prompt_function=olympiadbench_prompt_fn,
    hf_repo="zwhe99/simplerl-OlympiadBench",
    hf_subset="default",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metric=[olympiadbench_metric] + make_metrics_at_k(k_values, olympiadbench_metric),
    version=1, 
    num_samples=max(k_values)
)

# Add tasks to the table
TASKS_TABLE = []
TASKS_TABLE.append(aime24)
TASKS_TABLE.append(aime25)
TASKS_TABLE.append(math_500)
TASKS_TABLE.append(gpqa_diamond)
TASKS_TABLE.append(minervamath)
TASKS_TABLE.append(gsm8k)
TASKS_TABLE.append(amc23)
TASKS_TABLE.append(olympiadbench)

# MODULE LOGIC
if __name__ == "__main__":
    print([t["name"] for t in TASKS_TABLE])
    print(len(TASKS_TABLE))
