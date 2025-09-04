import re
from reasoning_gym.utils import extract_answer
from reasoning_gym.factory import DATASETS

def format_reward(completions, **kwargs):
        regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\n<answer>([\s\S]*?)<\/answer>$"
        matches = [re.match(regex, completion, flags=re.DOTALL) for completion in completions]
        return [1.0 if match else 0.0 for match in matches]

def accuracy_reward(completions, metadata, **kwargs):
    scores = []
    
    for completion, obj in zip(completions, metadata):
        dataset = kwargs['dataset'].data.datasets[obj["metadata"]["source_dataset"]]
        answer = extract_answer(completion)
        # Call score_answer on the instance, without explicitly passing self
        scores.append(dataset.score_answer(answer=answer, entry=obj))
    return scores