from lighteval.metrics.utils.metric_utils import (
    MetricCategory,
    MetricUseCase,
    SampleLevelMetric,
)
from collections import Counter
import logging
from typing import Callable, Literal, Sequence
from lighteval.metrics.utils.math_comparison import compare_gold_target
from lighteval.metrics.utils.metric_utils import MetricCategory, MetricUseCase, SampleLevelMetric
from lighteval.tasks.requests import Doc
from lighteval.utils.language import Language
from lighteval.utils.timeout import timeout
from lighteval.metrics.utils.extractive_match_utils import (  # noqa: F401
    ExprExtractionConfig,
    ExtractionTarget,
    extract_target_from_pred,
    get_extraction_regexes,
)
import numpy as np


logger = logging.getLogger(__name__)

def make_majority_at_k(k, threshold, base_sample_metric):
    def majority_at_k_fn(predictions, formatted_doc, **kwargs):
        results = [base_sample_metric.sample_level_fn(predictions=[predictions[i]], formatted_doc=formatted_doc, **kwargs) for i in range(k)]
        # Results are 0 or 1, so we can just sum them up
        majority_pred = sum(results) >= (k / 2)

        return float(majority_pred)

    return SampleLevelMetric(
        metric_name=f"maj@{k}",
        higher_is_better=True,
        category=MetricCategory.GENERATIVE_SAMPLING,
        use_case=MetricUseCase.MATH,
        sample_level_fn=majority_at_k_fn,
        corpus_level_fn=np.mean,
    )

def make_pass_at_k(k, threshold, base_sample_metric):
    def pass_at_k_fn(predictions, formatted_doc, **kwargs):
        top_k = predictions[:k]

        for pred in top_k:
            res = base_sample_metric.sample_level_fn(predictions=[pred], formatted_doc=formatted_doc, **kwargs)
            if res >= threshold:
                return 1.0
        return 0.0

    return SampleLevelMetric(
        metric_name=f"pass@{k}",
        higher_is_better=True,
        category=MetricCategory.GENERATIVE_SAMPLING,
        use_case=MetricUseCase.MATH,
        sample_level_fn=pass_at_k_fn,
        corpus_level_fn=np.mean,
    )

def make_accuracy_at_k(k, threshold, base_sample_metric):
    def accuracy_at_k_fn(predictions, formatted_doc, **kwargs):
        top_k = predictions[:k]
        correct = 0

        for pred in top_k:
            res = base_sample_metric.sample_level_fn(predictions=[pred], formatted_doc=formatted_doc, **kwargs)
            if res >= threshold:
                correct += 1

        return correct / k

    return SampleLevelMetric(
        metric_name=f"acc@{k}",
        higher_is_better=True,
        category=MetricCategory.GENERATIVE_SAMPLING,
        use_case=MetricUseCase.MATH,
        sample_level_fn=accuracy_at_k_fn,
        corpus_level_fn=np.mean,
    )


def make_metrics_at_k(k_values, base_sample_metric):
    return [
        make_majority_at_k(k, 0.99, base_sample_metric) for k in k_values
    ] + [
        make_pass_at_k(k, 0.99, base_sample_metric) for k in k_values
    ] + [
        make_accuracy_at_k(k, 0.99, base_sample_metric) for k in k_values
    ]




