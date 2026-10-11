"""Automated Benchmark CLI for Reference-Free Financial RAG Evaluation with LangSmith Experiment Sync.

Usage:
    python backend/evals/run_evals.py
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add backend directory to sys.path
BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from evals.ragas_evaluator import ragas_evaluator
from rag.retriever import financial_retriever
from agents.specialists.finance_qa import FinanceQAAgent
from observability.logger import get_custom_logger
from observability.langsmith_tracer import get_langsmith_client

logger = get_custom_logger(name=__name__)
DATASET_NAME = "financial-rag-benchmark"


def ensure_langsmith_dataset(client, queries: List[Dict[str, Any]]) -> str:
    """Ensures the benchmark dataset exists in LangSmith, registering queries if missing."""
    try:
        if not client.has_dataset(dataset_name=DATASET_NAME):
            dataset = client.create_dataset(
                dataset_name=DATASET_NAME,
                description="Financial AI Assistant RAG benchmark evaluating Faithfulness, Answer Relevance, and Context Relevance"
            )
            for item in queries:
                client.create_example(
                    inputs={
                        "query": item["query"],
                        "target_collection": item.get("target_collection", "Finance")
                    },
                    outputs={},  # Reference-free benchmark
                    dataset_id=dataset.id,
                    metadata={
                        "id": item["id"],
                        "domain": item.get("domain", "General Finance")
                    }
                )
            logger.info(f"Created LangSmith dataset '{DATASET_NAME}' with {len(queries)} examples.")
            print(f"📦 Registered new dataset '{DATASET_NAME}' on LangSmith with {len(queries)} benchmark examples.")
        else:
            logger.info(f"LangSmith dataset '{DATASET_NAME}' is ready.")
            print(f"📦 Using existing LangSmith dataset: '{DATASET_NAME}'")
        return DATASET_NAME
    except Exception as e:
        logger.warning(f"Failed to ensure LangSmith dataset: {e}")
        return ""


def run_benchmark():
    evals_file = Path(__file__).resolve().parent / "unlabeled_financial_queries.json"
    if not evals_file.exists():
        print(f"Error: Dataset not found at {evals_file}")
        sys.exit(1)

    with open(evals_file, "r", encoding="utf-8") as f:
        queries = json.load(f)

    print("\n" + "=" * 80)
    print("🚀 FINANCIAL RAG REFERENCE-FREE EVALUATION BENCHMARK")
    print("=" * 80)
    print(f"Total Test Queries: {len(queries)}")
    print("Metrics: Faithfulness (Groundedness), Answer Relevance, Context Relevance, Composite RAG Score")
    print("=" * 80 + "\n")

    qa_agent = FinanceQAAgent()
    langsmith_client = get_langsmith_client()

    results = []

    # Run LangSmith official Experiment if client is configured
    experiment_url = None
    if langsmith_client:
        try:
            from langsmith.evaluation import evaluate, EvaluationResult
            from langsmith.schemas import Run, Example

            ds_name = ensure_langsmith_dataset(langsmith_client, queries)
            if ds_name:
                print("\n🔗 Syncing evaluation run with LangSmith 'Datasets & Experiments' dashboard...")

                def predict_rag_pipeline(inputs: dict) -> dict:
                    q_text = inputs["query"]
                    col_name = inputs.get("target_collection", "Finance")
                    docs = financial_retriever.retrieve_sync(
                        query=q_text,
                        collection_name=col_name,
                        top_k=4
                    )
                    contexts = [d.page_content for d in docs]
                    agent_out = qa_agent.run(q_text)
                    response_text = agent_out.get("response", "")
                    scores = ragas_evaluator.evaluate_single_turn(
                        query=q_text,
                        contexts=contexts,
                        response=response_text
                    )
                    return {
                        "response": response_text,
                        "contexts": contexts,
                        "citations": agent_out.get("citations", []),
                        "faithfulness": scores["faithfulness"],
                        "answer_relevance": scores["answer_relevance"],
                        "context_relevance": scores["context_relevance"],
                        "overall_rag_score": scores["overall_rag_score"]
                    }

                def eval_faithfulness(run: Run, example: Example) -> EvaluationResult:
                    return EvaluationResult(key="faithfulness", score=run.outputs.get("faithfulness", 0.0))

                def eval_answer_relevance(run: Run, example: Example) -> EvaluationResult:
                    return EvaluationResult(key="answer_relevance", score=run.outputs.get("answer_relevance", 0.0))

                def eval_context_relevance(run: Run, example: Example) -> EvaluationResult:
                    return EvaluationResult(key="context_relevance", score=run.outputs.get("context_relevance", 0.0))

                def eval_overall_score(run: Run, example: Example) -> EvaluationResult:
                    return EvaluationResult(key="overall_rag_score", score=run.outputs.get("overall_rag_score", 0.0))

                exp_res = evaluate(
                    predict_rag_pipeline,
                    data=ds_name,
                    evaluators=[
                        eval_faithfulness,
                        eval_answer_relevance,
                        eval_context_relevance,
                        eval_overall_score
                    ],
                    experiment_prefix="financial-rag-benchmark",
                    description="Reference-free RAG evaluation evaluating Groundedness and Relevance",
                    client=langsmith_client,
                    max_concurrency=1
                )
                experiment_url = getattr(exp_res, "url", None)

                # Collect row-level scores from experiment results for local display
                for row in exp_res:
                    row_run = getattr(row, "run", None)
                    row_ex = getattr(row, "example", None)
                    if row_run and row_ex:
                        q_id = row_ex.metadata.get("id", "query") if row_ex.metadata else "query"
                        q_txt = row_ex.inputs.get("query", "")
                        outs = row_run.outputs or {}
                        results.append({
                            "id": q_id,
                            "query": q_txt,
                            "faithfulness": outs.get("faithfulness", 0.0),
                            "answer_relevance": outs.get("answer_relevance", 0.0),
                            "context_relevance": outs.get("context_relevance", 0.0),
                            "overall_rag_score": outs.get("overall_rag_score", 0.0),
                        })
        except Exception as e:
            logger.warning(f"LangSmith experiment execution encountered an error: {e}. Falling back to local execution.")
            print(f"⚠️ LangSmith evaluate() failed: {e}. Running local benchmark...")

    # Fallback to local loop if LangSmith was not available or results were not collected
    if not results:
        for item in queries:
            qid = item["id"]
            q_text = item["query"]
            col_name = item.get("target_collection", "Finance")

            print(f"[{qid}] Query: {q_text[:70]}...")

            docs = financial_retriever.retrieve_sync(
                query=q_text,
                collection_name=col_name,
                top_k=4
            )
            contexts = [d.page_content for d in docs]
            agent_out = qa_agent.run(q_text)
            response_text = agent_out.get("response", "")

            scores = ragas_evaluator.evaluate_single_turn(
                query=q_text,
                contexts=contexts,
                response=response_text
            )

            results.append({
                "id": qid,
                "query": q_text,
                "chunks_retrieved": len(contexts),
                **scores
            })

            print(
                f"      -> Faithfulness: {scores['faithfulness']:.2f} | "
                f"Answer Rel: {scores['answer_relevance']:.2f} | "
                f"Context Rel: {scores['context_relevance']:.2f} | "
                f"Overall: {scores['overall_rag_score']:.2f}"
            )

    # Summary Statistics
    if results:
        avg_faithfulness = sum(r["faithfulness"] for r in results) / len(results)
        avg_answer_rel = sum(r["answer_relevance"] for r in results) / len(results)
        avg_context_rel = sum(r["context_relevance"] for r in results) / len(results)
        avg_overall = sum(r["overall_rag_score"] for r in results) / len(results)

        print("\n" + "=" * 80)
        print("📊 BENCHMARK SUMMARY REPORT")
        print("=" * 80)
        print(f"Mean Faithfulness (Groundedness): {avg_faithfulness * 100:.1f}% (Target: >= 85%)")
        print(f"Mean Answer Relevance:           {avg_answer_rel * 100:.1f}% (Target: >= 80%)")
        print(f"Mean Context Relevance:          {avg_context_rel * 100:.1f}% (Target: >= 75%)")
        print(f"Overall Composite RAG Score:     {avg_overall * 100:.1f}%")
        print("=" * 80 + "\n")

    if experiment_url:
        print(f"✅ View complete evaluation scorecard on LangSmith:\n   👉 {experiment_url}\n")
    elif langsmith_client:
        print("✅ Results synchronized with active LangSmith project telemetry.")
    else:
        print("ℹ️ Note: Set LANGSMITH_API_KEY in .env to stream evaluation experiments to LangSmith.")


if __name__ == "__main__":
    run_benchmark()
