from lighteval.metrics.utils.metric_utils import (
    MetricCategory,
    MetricUseCase,
    SampleLevelMetric,
)
from collections import Counter
from itertools import groupby
import re
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
    extract_match,
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


def multilingual_extractive_match_metric_robust(
    language: Language = Language.ENGLISH,
    gold_extraction_target: Sequence[ExtractionTarget] = (ExprExtractionConfig(),),
    pred_extraction_target: Sequence[ExtractionTarget] = (ExprExtractionConfig(),),
    aggregation_function: Callable[[list[float]], float] = max,
    fallback_mode: Literal["no_fallback", "first_match"] = "first_match",
    extraction_mode: Literal["first_match", "any_match"] = "any_match",
    precision: int = 6,
    timeout_seconds: int = 5,
) -> SampleLevelMetric:
    """Creates a language-aware extractive match metric that extracts answers from the model's output.

    Known issues:
    - If the task is to simplify an expression, the metric might overestimate the accuracy. This is because if the model doesn't output any anchor for the extraction (e.g final answer is..),
        it's possible that the the extracted prediction will be the expression to simplify. Because we do simplifications ourselves, it can thus happen that sympy will correctly simplify the expression,
        thus it will match gold, despite model not doing anything. PRs to fix this are welcome.

    - There is currently no StringExtractionConfig, so if the gold is \boxed{\text{Friday}} and model outputs Friday it will not match, because nothing will be extracted.

    Args:
        language: Language
            The language of the samples.
        gold_extraction_target: Sequence[ExtractionTarget]
            Extraction targets to use for gold answers. Defaults to extracting simple math expressions.
        pred_extraction_target: Sequence[ExtractionTarget]
            Extraction targets to use for predictions. Defaults to extracting simple math expressions.
        aggregation_function: Callable[[list[float]], float]
            Function to aggregate scores when multiple golds/predictions are present. Defaults to max.
        fallback_mode: Literal["no_fallback", "first_match"]
            How to perform extraction. Defaults to "first_match".
            - "no_fallback": Only use first successfully parsed matches
            - "first_match": Use the first successfully parsed match + first match irregardless the parsing success
        extraction_mode: Literal["first_match", "any_match"]
            - "first_match": Only tries to extract the first regex match if it fails no other matches are tried
            - "any_match": Tries to extract any regex match

        precision: int
            Number of decimal places to use when comparing numerical values. Defaults to 6.
        timeout_seconds: int
            Timeout for the extraction (each attempt) and comparison. Defaults to 5.

    Returns:
        A sample level metric that extracts and compares mathematical expressions.

    """

    @timeout(2)
    def add_to_specifics_with_timeout(
        formatted_doc: Doc, extracted_predictions: list[list[str]], extracted_golds: list[list[str]]
    ) -> None:
        if formatted_doc.specific is None:
            formatted_doc.specific = {}

        formatted_doc.specific["extracted_predictions"] = [
            str(pred) for preds in extracted_predictions for pred in preds
        ]
        formatted_doc.specific["extracted_golds"] = [str(gold) for golds in extracted_golds for gold in golds]

    def sample_level_fn(golds: list[str], predictions: list[str], formatted_doc: Doc) -> float:
        gold_extraction_regexes = get_extraction_regexes(formatted_doc, gold_extraction_target, language)
        pred_extraction_regexes = get_extraction_regexes(formatted_doc, pred_extraction_target, language)

        extracted_predictions = [
            extract_target_from_pred_robust(pred, pred_extraction_regexes, fallback_mode, extraction_mode, timeout_seconds)
            for pred in predictions
        ]
        extracted_golds = [
            extract_target_from_pred_robust(gold, gold_extraction_regexes, fallback_mode, extraction_mode, timeout_seconds)
            for gold in golds
        ]

        # Assert on empty gold and warn on empty pred
        if any(len(g) == 0 for g in extracted_golds):
            logger.warning(f"We did not manage to extract a gold in the correct format. Gold: {golds}")
            extracted_golds = [[gold] for gold in golds]

        if all(len(p) == 0 for p in extracted_predictions):
            logger.warning(
                f"We did not manage to extract a prediction in the correct format. Gold: {golds}, Pred: {predictions}"
            )

        # We have to use timeout because the sypmy to str conversion can be very slow
        try:
            add_to_specifics_with_timeout(formatted_doc, extracted_predictions, extracted_golds)
        except Exception:  # noqa: E722
            logger.warning("Timeout when adding extracted predictions and golds to specific")

        return aggregation_function(
            [
                (
                    1.0
                    if any(
                        compare_gold_target(gold, pred, precision, timeout_seconds=timeout_seconds)
                        for gold in extracted_golds
                    )
                    else 0.0
                )
                for pred in extracted_predictions
            ]
        )

    return SampleLevelMetric(
        metric_name="extractive_match",
        sample_level_fn=sample_level_fn,
        category=MetricCategory.GENERATIVE,
        use_case=MetricUseCase.ACCURACY,
        corpus_level_fn=np.mean,
        higher_is_better=True,
    )


def extract_target_from_pred_robust(
    pred: str,
    target_res: list[tuple[list[tuple[re.Pattern[str], int]], ExtractionTarget]],
    fallback_mode: Literal["no_fallback", "first_match"] = "no_fallback",
    extraction_mode: Literal["first_match", "any_match"] = "any_match",
    timeout_seconds: int = 5,
):
    """Extracts targets from a prediction string using regex patterns.
    Returns first sucesffuly extracted match.

    Args:
        pred (str): The prediction string to extract from
        target_res (list[tuple[list[tuple[re.Pattern[str], int]], ExtractionTarget]]): List of regex patterns and their priorities for each target type
        fallback_mode (Literal["no_fallback", "first_match"], optional): How to handle extraction failures. Defaults to "no_fallback".
            - "no_fallback": Return only successfully parsed match
            - "first_match": Additionaly Include the first string match no matter how parsing finished
        extraction_mode (Literal["first_match", "any_match"], optional): How to handle extraction failures. Defaults to "any_match".
            - "first_match": Only tries to extract the first match
            - "any_match": Tries to extract any match
        timeout_seconds (int, optional): Maximum time in seconds to spend parsing each expression. Defaults to 5.

    Returns:
        list: List of extracted predictions, with first fallbac string appended if fallback_mode is "first_match"
    """
    extracted_predictions = []
    fallbacks = []

    # Get all patterns and sort by priority
    all_patterns = [
        (pattern, target_type, priority)
        for target_patterns, target_type in target_res
        for pattern, priority in target_patterns
    ]
    match_found = False

    # Group patterns by priority using itertools.groupby
    for _, patterns_group in groupby(sorted(all_patterns, key=lambda x: x[2]), key=lambda x: x[2]):
        # Find all matches for each pattern in this priority group
        matches_with_pos = (
            (match, match.start(), match.end(), target_type)
            for pattern, target_type, _ in patterns_group
            for match in pattern.finditer(pred)
        )

        # Sort matches by end position (rightmost first) and then by start position (leftmost first)
        matches_with_pos = sorted(matches_with_pos, key=lambda x: (x[2], -x[1]), reverse=True)

        # Try to extract from each match, starting from rightmost
        for match, _, _, target_type in matches_with_pos:
            try:
                extracted_match, str_fallback = extract_match(match, target_type, timeout_seconds)
            except Exception:
                continue

            match_found = True

            if str_fallback:
                fallbacks.append(str_fallback)

            if extracted_match is not None:
                extracted_predictions.append(extracted_match)
                break

            if extraction_mode == "first_match":
                break

        # If we found something and we're in first_match mode, stop processing other priorities
        if extracted_predictions or (match_found and extraction_mode == "first_match"):
            break

    if fallback_mode == "first_match" and fallbacks:
        extracted_predictions += [fallbacks[0]]

    return extracted_predictions


