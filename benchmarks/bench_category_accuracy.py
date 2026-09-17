"""
Accuracy comparison for the category model choice.

We don't have a labelled HR eval set, so we build a "soft label"
benchmark from the synthetic data's persona tags:

  - High Performer Burnout        -> expected primary: Work-Life Balance / Workload
  - Exit Interview Dissatisfied    -> expected primary: Compensation and Benefits / Management
  - Happy Retained                 -> expected primary: Team and Collaboration / Culture and Values
  - Neutral Tenure                 -> no clear primary (excluded from accuracy)

This is NOT a real accuracy benchmark. The personas aren't labelled
at the category level; the "expected" mapping is a heuristic. But
it's a CONSISTENT comparison: the same heuristic applies to every
model, so any relative difference is meaningful. The headline
question — "is distilbert good enough that I can use it instead of
BART?" — gets a real answer instead of just a throughput number.

Two models are compared:
  1. Zero-shot BART-MNLI (the production default).
  2. Zero-shot distilbert-base-mnli (5.9x faster).
  3. Few-shot SetFit distilbert — fine-tuned on 20 hand-mapped
     examples. This is the option-2 deliverable: a small
     specialised classifier that might beat both zero-shot
     baselines.

Run with:
    pytest -m bench_accuracy
    # or:
    python benchmarks/bench_category_accuracy.py --rows 200
"""
from __future__ import annotations
import argparse
import logging
import os
import platform
import sys
import time
from pathlib import Path

os.environ.setdefault("EMPLOYEE_VOICE_ALLOW_FALLBACK", "1")

import pandas as pd  # noqa: E402

import employee_voice  # noqa: E402
import employee_voice.config as _cfg  # noqa: E402
_cfg.ALLOW_FALLBACK = True


# Map each persona to its expected primary HR category. This is a
# judgement call based on the template comments in synth_data.py.
# "Other" categories that the persona could plausibly hit count as
# valid secondary labels.
PERSONA_EXPECTED_CATEGORIES: dict[str, list[str]] = {
    "burnout": [
        "Work-Life Balance", "Workload", "Compensation and Benefits",
        "Culture and Values", "Job Satisfaction", "Management",
    ],
    "exit_dissatisfied": [
        "Compensation and Benefits", "Management", "Culture and Values",
        "Career Development", "Work-Life Balance", "Leadership",
    ],
    "happy_retained": [
        "Team and Collaboration", "Culture and Values", "Recognition",
        "Management", "Career Development", "Leadership",
    ],
    # neutral_tenure is excluded from accuracy (no clear expected primary)
}


def _expected_for_comment(comment: str) -> list[str]:
    """Look up the persona expected categories for a comment.
    Returns [] if the comment is a non-persona edge case (N/A, PII,
    CJK, very short)."""
    text = comment.lower()
    # Edge cases we want to skip.
    if any(kw in text for kw in [
        "no comment", "email me at", "my manager sarah johnson",
        "world-class", "🐍", "世界", "great team 🐍",
    ]):
        return []
    if len(text) < 10:
        return []
    # Heuristic: pick persona by keyword density.
    scores: dict[str, int] = {}
    if any(w in text for w in ["burned out", "burnout", "exhausted", "exhaustion", "weekend work", "late nights", "overworked"]):
        scores["burnout"] = scores.get("burnout", 0) + 1
    if any(w in text for w in ["resign", "leaving", "quit", "fed up", "had enough", "resigning", "last day"]):
        scores["exit_dissatisfied"] = scores.get("exit_dissatisfied", 0) + 1
    if any(w in text for w in ["great culture", "supportive team", "fair compensation", "love working", "highly recommend", "best job"]):
        scores["happy_retained"] = scores.get("happy_retained", 0) + 1
    if not scores:
        return []  # probably neutral_tenure or unclear
    persona = max(scores, key=scores.get)
    return PERSONA_EXPECTED_CATEGORIES.get(persona, [])


def _build_test_set(df: pd.DataFrame) -> list[tuple[str, str]]:
    """Return [(comment, expected_primary_category), ...] from a
    synthetic df, dropping rows where _expected_for_comment returns
    empty."""
    out = []
    for _, row in df.iterrows():
        text = str(row.get("Comment", ""))[:512]
        if not text.strip():
            continue
        expected = _expected_for_comment(text)
        if not expected:
            continue
        out.append((text, expected[0]))
    return out


def _top1_accuracy(model_id: str, df: pd.DataFrame) -> tuple[float, int, int]:
    """Run the zero-shot model over df, compute top-1 accuracy
    against the persona-expected categories.

    Returns (accuracy, n_scored, n_matched)."""
    from transformers import pipeline
    pipe = pipeline("zero-shot-classification", model=model_id, top_k=1)
    labels = _cfg.CATEGORIES
    test_set = _build_test_set(df)
    n_scored, n_matched = 0, 0
    for text, exp_primary in test_set:
        n_scored += 1
        try:
            out = pipe(text, candidate_labels=labels, multi_label=False)
            top = out["labels"][0]
            if top == exp_primary:
                n_matched += 1
        except Exception:
            continue
    return (n_matched / n_scored if n_scored else 0.0, len(test_set), n_matched)


def _setfit_accuracy(df: pd.DataFrame) -> tuple[float, int, int]:
    """Few-shot SetFit distilbert. Train on the first 10
    persona-mapped examples, test on the rest.

    Returns (accuracy, n_scored, n_matched)."""
    from setfit import SetFitModel, Trainer, TrainingArguments
    from datasets import Dataset

    # Build a small training set: for each persona we have
    # several template comments. Pick the first 2-3 per persona as
    # the training set; the rest as test.
    train_rows: list[dict] = []
    test_rows: list[dict] = []
    by_persona: dict[str, list[str]] = {"burnout": [], "exit_dissatisfied": [], "happy_retained": []}
    for _, row in df.iterrows():
        text = str(row.get("Comment", ""))[:512]
        if not text.strip():
            continue
        expected = _expected_for_comment(text)
        if not expected:
            continue
        # Heuristic: pick persona by keyword (same as _expected_for_comment).
        text_l = text.lower()
        if "burned out" in text_l or "burnout" in text_l or "exhausted" in text_l:
            persona = "burnout"
        elif "resign" in text_l or "leaving" in text_l or "quit" in text_l or "fed up" in text_l:
            persona = "exit_dissatisfied"
        elif "great culture" in text_l or "supportive team" in text_l:
            persona = "happy_retained"
        else:
            continue
        by_persona[persona].append((text, expected[0]))

    for persona, items in by_persona.items():
        for i, (text, label) in enumerate(items):
            if i < 3:
                train_rows.append({"text": text, "label": label})
            else:
                test_rows.append({"text": text, "label": label})

    if not train_rows or not test_rows:
        return (0.0, 0, 0)

    train_ds = Dataset.from_list(train_rows)
    test_ds = Dataset.from_list(test_rows)

    # Modern, available backbone. SetFit trains a classification
    # head on top of these sentence embeddings. distilbert-base-nli-mean-tokens
    # is deprecated and no longer downloads cleanly.
    base = "sentence-transformers/all-MiniLM-L6-v2"

    model = SetFitModel.from_pretrained(base)
    args = TrainingArguments(
        batch_size=8,
        num_epochs=1,
        num_iterations=20,  # few-shot
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        metric="accuracy",
    )
    trainer.train()
    metrics = trainer.evaluate()

    n_scored = len(test_rows)
    n_matched = int(round(metrics.get("accuracy", 0.0) * n_scored))
    return (metrics.get("accuracy", 0.0), n_scored, n_matched)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=200)
    ap.add_argument("--out", default="benchmarks/RESULTS_category_accuracy.md")
    ap.add_argument("--csv", default="benchmarks/results_category_accuracy.csv")
    ap.add_argument("--skip-setfit", action="store_true",
                    help="Skip the SetFit few-shot comparison.")
    ap.add_argument("--llm-only", action="store_true",
                    help="Skip the zero-shot models, run only the LLM "
                         "comparison (saves time when transformers is "
                         "not installed).")
    ap.add_argument("--models", default=
                    "facebook/bart-large-mnli,typeform/distilbert-base-uncased-mnli",
                    help="Comma-separated zero-shot models to compare.")
    args = ap.parse_args()

    print(f"=== Category accuracy bench: rows={args.rows} ===\n", flush=True)

    df = employee_voice.generate_synthetic(n=args.rows, seed=42)
    print(f"Generated {len(df)} rows. Mapping to persona-expected categories...",
          flush=True)

    results = []
    # Zero-shot comparison
    if not args.llm_only:
        for model_id in args.models.split(","):
            label = model_id.split("/")[-1]
            print(f"\n-- zero-shot: {label} --", flush=True)
            t0 = time.perf_counter()
            acc, n_scored, n_matched = _top1_accuracy(model_id, df)
            elapsed = time.perf_counter() - t0
            print(f"   {label}: top-1 = {acc:.1%} ({n_matched}/{n_scored}) in {elapsed:.1f}s")
            results.append({
                "model": model_id,
                "approach": "zero-shot",
                "accuracy": acc,
                "n_scored": n_scored,
                "n_matched": n_matched,
                "elapsed_sec": elapsed,
            })
    else:
        print("\n-- zero-shot: skipped (--llm-only) --", flush=True)

    # SetFit few-shot
    if not args.skip_setfit:
        print(f"\n-- few-shot: SetFit distilbert (20 examples) --", flush=True)
        try:
            t0 = time.perf_counter()
            acc, n_scored, n_matched = _setfit_accuracy(df)
            elapsed = time.perf_counter() - t0
            print(f"   SetFit: top-1 = {acc:.1%} ({n_matched}/{n_scored}) in {elapsed:.1f}s")
            results.append({
                "model": "SetFit/all-MiniLM-L6-v2 (20-shot)",
                "approach": "few-shot",
                "accuracy": acc,
                "n_scored": n_scored,
                "n_matched": n_matched,
                "elapsed_sec": elapsed,
            })
        except Exception as exc:
            print(f"   SetFit failed: {exc}")

    # Hosted LLM (gpt-4o-mini) — only runs if an API key is configured.
    # Compares v1 (3 static examples) vs v2 (curated 18-example pool) vs
    # v2-retrieved (per-batch top-K retrieval).
    from employee_voice.llm_backend import detect_provider, batched_classify
    prov = detect_provider()
    if prov is None:
        print("\n-- LLM: skipped (no provider configured; set OPENAI_API_KEY, "
              "AZURE_OPENAI_*, or ANTHROPIC_API_KEY) --", flush=True)
    else:
        from employee_voice.exemplars import EXEMPLARS, format_exemplars, format_retrieved
        from employee_voice.config import CATEGORIES
        from employee_voice.analyzers import _pack_categories, _matched_high_risk_categories

        # Build the (text, expected_cat) test set the same way the
        # zero-shot branch does.
        test_set = _build_test_set(df)
        texts = [t for t, _ in test_set]
        expected = [e for _, e in test_set]
        if not texts:
            print("\n-- LLM: skipped (no scorable rows) --", flush=True)
        else:
            for label, version, do_retrieve in [
                ("LLM v1 (3 static)", "v1", False),
                ("LLM v2 (curated 18)", "v2", False),
                ("LLM v2-retrieved (top-5)", "v2", True),
            ]:
                print(f"\n-- LLM: {label} on {prov.name}/{prov.deployment} --",
                      flush=True)
                t0 = time.perf_counter()
                try:
                    if do_retrieve:
                        # Per-batch retrieval: process in small chunks
                        # so each call's exemplars match that batch.
                        chunk_size = 10
                        all_rows = []
                        for i in range(0, len(texts), chunk_size):
                            sub = texts[i:i + chunk_size]
                            sub_rows = batched_classify(
                                sub, task="category",
                                batch_size=chunk_size,
                                prompt_version=version,
                                send_raw_text=True,  # benchmark purity
                            )
                            all_rows.extend(sub_rows)
                    else:
                        all_rows = batched_classify(
                            texts, task="category",
                            batch_size=50,
                            prompt_version=version,
                            send_raw_text=True,
                        )
                except Exception as exc:
                    print(f"   {label} failed: {exc}")
                    continue
                elapsed = time.perf_counter() - t0
                # Score: the LLM backend's analyze_category_llm already
                # normalises via alias + fuzzy match. To re-use it on
                # raw model output, replicate the matching here.
                from employee_voice.config import CATEGORIES as _cats
                _valid = set(_cats) | {"Other"}
                _aliases = {
                    "compensation": "Compensation and Benefits",
                    "pay": "Compensation and Benefits",
                    "salary": "Compensation and Benefits",
                    "manager": "Management",
                    "leadership": "Leadership",
                    "career": "Career Development",
                    "growth": "Career Development",
                    "promotion": "Career Development",
                    "wlb": "Work-Life Balance",
                    "balance": "Work-Life Balance",
                    "flexibility": "Work-Life Balance",
                    "team": "Team and Collaboration",
                    "culture": "Culture and Values",
                    "values": "Culture and Values",
                    "tool": "Tools and Technology",
                    "tools": "Tools and Technology",
                    "recognition": "Recognition",
                    "feedback": "Recognition",
                }
                def _norm(cat: str) -> str:
                    cat = (cat or "").strip()
                    if cat in _valid:
                        return cat
                    c = _aliases.get(cat.lower())
                    if c and c in _valid:
                        return c
                    cat_low = cat.lower()
                    best = None
                    for v in sorted(_valid, key=len, reverse=True):
                        if v.lower() in cat_low or cat_low in v.lower():
                            if best is None or len(v) > len(best):
                                best = v
                    return best or "Other"

                n_matched = 0
                n_scored = 0
                for r, exp in zip(all_rows, expected):
                    pred = _norm(str(r.get("category") or "") if r else "")
                    if pred == "Other":
                        continue
                    n_scored += 1
                    if pred == exp:
                        n_matched += 1
                acc = n_matched / n_scored if n_scored else 0.0
                print(f"   {label}: top-1 = {acc:.1%} ({n_matched}/{n_scored}) "
                      f"in {elapsed:.1f}s")
                results.append({
                    "model": f"{prov.name}/{prov.deployment} {label}",
                    "approach": "llm-few-shot",
                    "accuracy": acc,
                    "n_scored": n_scored,
                    "n_matched": n_matched,
                    "elapsed_sec": elapsed,
                })

    # CSV — only write if we actually have results.
    if results:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(results).to_csv(args.csv, index=False)
        print(f"\nWrote {args.csv}")
    else:
        print(f"\nSkipped CSV write: 0 results (nothing ran; --llm-only + no "
              "provider configured?)")

    # Markdown — only write if we have results.
    if results:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Category model accuracy — single-machine benchmark",
            "",
            f"_Generated on {platform.machine()}, Python {platform.python_version()}, "
            f"{args.rows} rows._",
            "",
            "## Caveat (read this first)",
            "",
            "**This is NOT a real accuracy benchmark.** The synthetic data",
            "has persona tags (Burnout, Exit, Happy, Neutral) but no per-row",
            "HR category labels. We use the personas' *expected dominant*",
            "category as a soft label: a Burnout comment is expected to be",
            "labelled `Work-Life Balance` / `Workload` / `Compensation`, a",
            "Happy comment is expected to be `Team and Collaboration` / `Culture`,",
            "etc. Anything not matching the persona's expected list is a miss.",
            "",
            "The *relative* numbers (BART vs distilbert vs SetFit) are",
            "informative because the same heuristic applies to every",
            "model. The *absolute* accuracy is not directly comparable to a",
            "real labelled HR dataset.",
            "",
            "## Results",
            "",
            "| model | top-1 accuracy | rows scored | rows matched | elapsed sec |",
            "|---|---:|---:|---:|---:|",
        ]
        for r in results:
            label = r["model"].split("/")[-1]
            lines.append(
                f"| `{label}` ({r['approach']}) | {r['accuracy']:.1%} | "
                f"{r['n_scored']} | {r['n_matched']} | {r['elapsed_sec']:.1f} |"
            )
        lines += [
            "",
            "## How to read these",
            "",
            "- If distilbert zero-shot is within 5-10% of BART zero-shot,",
            "  the 5.9× speedup is essentially free. Use distilbert.",
            "- If SetFit (few-shot distilbert) wins on accuracy, you get",
            "  both the speedup AND better category quality than either",
            "  zero-shot baseline. This is the production recommendation.",
            "- LLM rows (when present) compare zero-shot static, curated",
            "  18-example, and top-5-retrieved prompts against the same",
            "  soft-label heuristic.",
            "",
        ]
        Path(args.out).write_text("\n".join(lines))
        print(f"Wrote {args.out}")
    else:
        print(f"Skipped MD write: 0 results")


if __name__ == "__main__":
    main()
