"""Morning diagnostic scoring; no model loading or formal evaluation."""
import json
import math
import re
from pathlib import Path

from vendor.mmf_m4c_evaluators import EvalAIAnswerProcessor, TextVQAAccuracyEvaluator

ROOT = Path(__file__).resolve().parents[1]


def textvqa_score(prediction, answers):
    return TextVQAAccuracyEvaluator().eval_pred_list(
        [{"pred_answer": prediction, "gt_answers": answers}]
    )


def extract_choice(text, choices):
    """Conservative, label-blind parser. Unparseable/ambiguous -> None."""
    text = text.strip()
    # Accept a bare option letter, or an explicit 'Answer: B' declaration.
    bare = re.fullmatch(r"\(?([A-Z])\)?[.)]?", text)
    declared = re.findall(
        r"(?:the\s+)?(?:correct\s+)?answer\s*(?:is\s*|:\s*)\(?([A-Z])\)?(?=[\s.):]|$)",
        text, flags=re.IGNORECASE,
    )
    # Some real outputs include both a letter and the full option text.
    # Require agreement with that option's text; never consult the label.
    prefixed = re.fullmatch(r"\(?([A-Z])\)?[.)]\s+(.+)", text, flags=re.IGNORECASE)
    if prefixed:
        index = ord(prefixed.group(1).upper()) - ord("A")
        option_text = prefixed.group(2).strip().rstrip(".! ").casefold()
        if 0 <= index < len(choices) and option_text == choices[index].strip().rstrip(".! ").casefold():
            return index
        return None
    letters = [bare.group(1)] if bare else [s.upper() for s in declared]
    if letters:
        indices = {ord(s) - ord("A") for s in letters}
        if len(indices) == 1:
            index = indices.pop()
            return index if 0 <= index < len(choices) else None
        return None
    # Exact option-text match only; no label-informed correction or fuzzy match.
    matches = [i for i, choice in enumerate(choices) if text.casefold() == choice.strip().casefold()]
    return matches[0] if len(matches) == 1 else None


def scienceqa_accuracy(predictions, labels):
    """Upstream evaluate_acc.py equality and denominator, adapted to a subset.

    Fraction in [0, 1]; multiply by 100 for upstream percentage display.
    None predictions count as wrong and remain in the denominator.
    """
    if not predictions or len(predictions) != len(labels):
        raise ValueError("Require a nonempty pair of equally sized lists")
    return sum(p is not None and p == y for p, y in zip(predictions, labels)) / len(labels)


def score_response(dataset, raw, sample):
    """Shared diagnostic/batch adapter; never passes labels to extraction."""
    empty = not raw.strip()
    if dataset == "textvqa":
        prediction = raw.strip()
        return dict(prediction=prediction, normalized_prediction=EvalAIAnswerProcessor()(prediction),
                    reference_answers=sample["answers"],
                    score=0.0 if empty else textvqa_score(prediction, sample["answers"]),
                    parse_status="empty_output" if empty else "ok",
                    extraction_policy="outer whitespace strip followed by unchanged MMF normalization")
    if dataset != "scienceqa":
        raise ValueError(dataset)
    prediction = extract_choice(raw, sample["choices"])
    return dict(prediction=prediction, label=sample["answer"],
                prediction_choice=sample["choices"][prediction] if prediction is not None else None,
                score=scienceqa_accuracy([prediction], [sample["answer"]]),
                parse_status="empty_output" if empty else "parse_failure" if prediction is None else "ok",
                extraction_policy="label-blind v2 option extraction; None counts wrong")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checks = []
    for count, expected in [(0, 0), (1, .3), (2, .6), (3, .9), (4, 1), (10, 1)]:
        score = textvqa_score("yes", ["yes"] * count + ["no"] * (10 - count))
        assert math.isclose(score, expected, abs_tol=1e-12)
        checks.append({"yes_count": count, "prediction": "yes", "actual": score, "expected": expected})
    assert textvqa_score("wrong", ["yes"] * 10) == 0
    normalizer = EvalAIAnswerProcessor()
    normalization = {s: normalizer(s) for s in ["The TWO!", "YES.", "dont", "1,000"]}
    assert normalizer("The TWO!") == "2"
    assert textvqa_score("The TWO!", ["2"] * 10) == 1
    score_cases = []
    for predictions, labels, expected in [([0, 1], [0, 1], 1), ([1, 0], [0, 1], 0), ([0, None], [0, 1], .5)]:
        actual = scienceqa_accuracy(predictions, labels)
        assert actual == expected
        score_cases.append(dict(predictions=predictions, labels=labels, actual=actual, expected=expected))
    choices = ["red", "blue", "green"]
    parser_cases = [("B", 1), ("(B)", 1), ("Answer: B", 1), ("The answer is B.", 1),
                    ("B. blue.", 1), ("A. blue.", None), ("B. blue. Answer: A.", None),
                    ("blue", 1), ("I cannot tell", None), ("Answer: D", None),
                    ("Answer: A. Answer: B.", None)]
    for text, expected in parser_cases:
        assert extract_choice(text, choices) == expected, text
    result = {"status": "passed", "textvqa": checks, "obviously_wrong": 0,
              "normalization": normalization, "scienceqa": score_cases,
              "parser_cases": [{"raw": t, "actual": extract_choice(t, choices), "expected": e} for t, e in parser_cases],
              "unparseable_policy": "Preserve raw output; None counts as incorrect, denominator unchanged"}
    output = args.output_dir / "scoring_checks.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
