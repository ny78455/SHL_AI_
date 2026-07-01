"""
Evaluation harness — replays C1–C10 conversation traces against the live API
and computes Recall@10 per trace.

Usage:
    # Start the server first
    uvicorn src.main:app --reload &

    # Run harness
    python -m tests.eval.harness --url http://localhost:8000

Outputs:
- Per-trace Recall@10 scores
- Overall mean Recall@10
- Failure details for any URL mismatches
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import httpx

TRACES_PATH = Path(__file__).parent / "traces" / "c1_c10.json"


def load_traces() -> list[dict[str, Any]]:
    return json.loads(TRACES_PATH.read_text(encoding="utf-8"))


def replay_trace(
    trace: dict[str, Any],
    base_url: str,
    client: httpx.Client,
) -> dict[str, Any]:
    """
    Replay a single trace: send all user turns (with full history each time)
    and collect the final committed recommendations.

    Returns a result dict with recall@10 score.
    """
    trace_id = trace["id"]
    final_shortlist_urls: list[str] = trace["final_shortlist_urls"]

    # Build message history progressively
    history: list[dict[str, str]] = []
    last_recommendations: list[str] = []

    # Filter to turns present in the fixture (some traces pre-load history)
    turns = trace["turns"]

    # Feed in turn-by-turn
    user_turns = [t for t in turns if t["role"] == "user"]

    for turn in turns:
        history.append({"role": turn["role"], "content": turn["content"]})

        if turn["role"] != "user":
            continue  # Only POST after each user message

        try:
            response = client.post(
                f"{base_url}/chat",
                json={"messages": history},
                timeout=35.0,
            )
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            return {
                "trace_id": trace_id,
                "error": str(exc),
                "recall_at_10": 0.0,
                "returned_urls": [],
                "expected_urls": final_shortlist_urls,
            }

        # Add assistant response to history
        assistant_reply = data.get("reply", "")
        history.append({"role": "assistant", "content": json.dumps(data)})

        recs = data.get("recommendations") or []
        if recs:
            last_recommendations = [r["url"] for r in recs]

        if data.get("end_of_conversation"):
            break

    # Compute Recall@10
    predicted_set = set(last_recommendations[:10])
    expected_set = set(final_shortlist_urls)

    if not expected_set:
        recall = 1.0
    else:
        hits = len(predicted_set & expected_set)
        recall = hits / len(expected_set)

    return {
        "trace_id": trace_id,
        "recall_at_10": recall,
        "hits": len(predicted_set & expected_set),
        "expected_count": len(expected_set),
        "returned_urls": list(predicted_set),
        "expected_urls": final_shortlist_urls,
        "missed_urls": list(expected_set - predicted_set),
    }


def run_harness(base_url: str) -> None:
    traces = load_traces()
    print(f"\n{'='*60}")
    print(f"  SHL Assessment Recommender — Eval Harness")
    print(f"  Target: {base_url}")
    print(f"  Traces: {len(traces)}")
    print(f"{'='*60}\n")

    results = []
    with httpx.Client() as client:
        # Health check first
        try:
            r = client.get(f"{base_url}/health", timeout=10.0)
            assert r.json()["status"] == "ok", "Health check failed"
            print("✓ /health OK\n")
        except Exception as exc:
            print(f"✗ Health check failed: {exc}")
            sys.exit(1)

        for trace in traces:
            result = replay_trace(trace, base_url, client)
            results.append(result)

            score = result.get("recall_at_10", 0.0)
            hits = result.get("hits", 0)
            total = result.get("expected_count", 0)
            icon = "✓" if score >= 0.7 else ("~" if score >= 0.5 else "✗")
            print(f"  {icon} {result['trace_id']:>4}  Recall@10={score:.2f}  ({hits}/{total})")

            if result.get("missed_urls"):
                for url in result["missed_urls"]:
                    print(f"        MISS: {url}")

    # Summary
    valid = [r for r in results if "error" not in r]
    mean_recall = sum(r["recall_at_10"] for r in valid) / len(valid) if valid else 0.0

    print(f"\n{'='*60}")
    print(f"  Mean Recall@10 : {mean_recall:.3f}")
    print(f"  Traces passed  : {sum(1 for r in valid if r['recall_at_10'] >= 0.8)}/{len(valid)}")
    print(f"{'='*60}\n")

    if mean_recall < 0.6:
        print("⚠ Mean Recall@10 below 0.6 — check retrieval configuration.")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SHL Eval Harness")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    args = parser.parse_args()
    run_harness(args.url)
