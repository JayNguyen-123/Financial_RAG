"""LangSmith regression gate for CI/CD.

Runs the *production* RAGService over a golden dataset, grades each answer
with an LLM judge, and exits non-zero if accuracy is below the threshold.

Fixes vs. the original:
* It imported ``multimodal_rag_chain`` from src.main, which never existed ->
  ImportError on every CI run.
* ``run_on_dataset`` with a fixed ``project_name`` fails on the second run
  (project already exists); experiments now get unique names.
* The gate looked for a "correctness" feedback key, but the COT_QA evaluator
  emits a different key, so no scores were ever found -> always exit 1.
  Scores are now read directly from the experiment results.
* Target errors (exceptions, guardrail blocks) count as failures instead of
  being silently excluded from the average.

Usage:  python -m src.smoke_test
Env:    EVAL_DATASET, EVAL_TENANT (default "eval"), EVAL_ACCURACY_THRESHOLD (default 0.90),
        EVAL_MIN_EXAMPLES (default 5), EVAL_MAX_CONCURRENCY (default 2)
"""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Any, Dict

from pydantic import BaseModel, Field

from config.settings import get_settings
from src.logging_config import configure_logging

DATASET = os.getenv("EVAL_DATASET", "Multimodal_RAG_Image_Parsing_Regression_Suite")
THRESHOLD = float(os.getenv("EVAL_ACCURACY_THRESHOLD", "0.90"))
MIN_EXAMPLES = int(os.getenv("EVAL_MIN_EXAMPLES", "5"))
MAX_CONCURRENCY = int(os.getenv("EVAL_MAX_CONCURRENCY", "2"))
# Tenant whose namespace holds the golden documents. The eval runs as an admin
# of that tenant so ACLs don't hide golden documents.
EVAL_TENANT = os.getenv("EVAL_TENANT", "eval")

JUDGE_PROMPT = (
    "You are grading a financial question-answering system.\n"
    "Question: {question}\nReference answer: {reference}\nSystem answer: {prediction}\n\n"
    "Think step by step. The system answer is CORRECT only if every figure, unit, period "
    "and entity it states agrees with the reference, and it does not omit the key figure "
    "the question asks for. Extra correct context is fine; any wrong or invented number "
    "makes it INCORRECT."
)


class Grade(BaseModel):
    reasoning: str = Field(description="Step-by-step comparison against the reference")
    correct: bool


def build_correctness_evaluator(settings):
    from langchain_openai import ChatOpenAI

    judge = ChatOpenAI(
        model=settings.GENERATION_MODEL,
        temperature=0,
        api_key=settings.OPENAI_API_KEY.get_secret_value(),
        timeout=settings.OPENAI_TIMEOUT_SECONDS,
        max_retries=settings.OPENAI_MAX_RETRIES,
    ).with_structured_output(Grade)

    def correctness(run, example) -> Dict[str, Any]:
        prediction = (run.outputs or {}).get("output", "")
        if run.error or not prediction:
            return {"key": "correctness", "score": 0.0, "comment": f"target error: {run.error}"}
        reference = (example.outputs or {}).get("answer") or (example.outputs or {}).get("output", "")
        grade: Grade = judge.invoke(
            JUDGE_PROMPT.format(question=example.inputs["question"], reference=reference, prediction=prediction)
        )
        return {"key": "correctness", "score": 1.0 if grade.correct else 0.0, "comment": grade.reasoning}

    return correctness


async def run() -> int:
    settings = get_settings()
    configure_logging(settings.LOG_LEVEL, json_logs=False)
    if settings.LANGCHAIN_API_KEY is None:
        print("❌ LANGCHAIN_API_KEY is required for the smoke test.")
        return 1
    settings.export_langsmith_env()

    from langsmith import Client
    from langsmith.evaluation import aevaluate

    from src.rag import RAGService
    from src.tenancy import Principal

    client = Client(api_key=settings.LANGCHAIN_API_KEY.get_secret_value())
    if not client.has_dataset(dataset_name=DATASET):
        print(f"❌ Validation dataset '{DATASET}' not found in LangSmith.")
        return 1

    service = RAGService.from_settings(settings)
    principal = Principal(sub="ci-smoke-test", tenant=EVAL_TENANT, is_admin=True)

    async def target(inputs: Dict[str, Any]) -> Dict[str, Any]:
        result = await service.answer(inputs["question"], principal, trace_id="ci-smoke-test")
        return {"output": result["answer"]}

    sha = os.getenv("GITHUB_SHA", "local")[:7]
    print(f"🧪 Running evaluation on '{DATASET}' (threshold {THRESHOLD:.0%})...")
    results = await aevaluate(
        target,
        data=DATASET,
        evaluators=[build_correctness_evaluator(settings)],
        experiment_prefix=f"ci-smoke-{sha}",
        metadata={"git_sha": os.getenv("GITHUB_SHA", "local"), "model": settings.GENERATION_MODEL},
        max_concurrency=MAX_CONCURRENCY,
        client=client,
    )

    scores = []
    async for row in results:
        eval_results = (row.get("evaluation_results") or {}).get("results", [])
        score = next((r.score for r in eval_results if r.key == "correctness"), None)
        scores.append(float(score) if score is not None else 0.0)

    if len(scores) < MIN_EXAMPLES:
        print(f"❌ Only {len(scores)} examples evaluated (minimum {MIN_EXAMPLES}).")
        return 1

    accuracy = sum(scores) / len(scores)
    print(f"📊 Accuracy: {accuracy:.2%} over {len(scores)} examples")
    if accuracy < THRESHOLD:
        print(f"❌ Deployment blocked: accuracy below {THRESHOLD:.0%}.")
        return 1
    print("✅ Smoke test passed.")
    return 0


def main() -> None:
    sys.exit(asyncio.run(run()))


if __name__ == "__main__":
    main()
