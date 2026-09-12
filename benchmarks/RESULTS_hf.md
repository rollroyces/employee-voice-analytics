# HuggingFace backend — single-machine benchmark

_Numbers produced by `benchmarks/bench_hf.py` (one subprocess per layer, warmup + measure pattern, child prints a parseable `MEASURE` line). Each layer runs in its own subprocess so the warm cache from one layer can't contaminate the next._

| layer | rows | elapsed sec | **rows/sec** |
|---|---:|---:|---:|
| `sentiment` (cardiffnlp/twitter-roberta-base-sentiment-latest) | 100 | 4.50 | **22.2** |
| `category` (facebook/bart-large-mnli, zero-shot) | 100 | 66.42 | **1.5** |
| `emotion` (SamLowe/roberta-base-go_emotions) | 100 | 4.13 | **24.2** |

## Reading these numbers

- **Apple M-series arm64** with MPS GPU, Python 3.14.4. CPU-only would be ~5× slower. A discrete GPU cluster would be ~10-50× faster again.
- **Sentiment + Emotion** are tractable on a single machine at ~22-24 rows/sec — that's roughly **12 hours for 1M rows** on a single CPU. A 4-core Spark executor with the same models gets you 8-16× throughput via `pandas_udf(SCALAR_ITER)`.
- **Category is the bottleneck** at 1.5 rows/sec because BART-MNLI is a 400M-parameter zero-shot model — each forward pass scores the text against all 18 HR categories. **1M rows would take ~7.7 days on a single machine.** This is not a viable path without parallelism. The right answers are:
  1. **Use the lexicon backend for category** on a CPU. 30k rows/sec = 33 seconds for 1M rows. The lexicon's accuracy is lower but predictable.
  2. **Run HF category on a Spark cluster** with `analyze_spark()`. The `pandas_udf(SCALAR_ITER)` path parallelises per-row inference across executors. On a 10-node cluster expect ~150 rows/sec *per node* = 1.5k rows/sec aggregate = ~11 minutes for 1M rows.
  3. **Distill BART** to a smaller model (e.g. `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`) for a 10× speedup at modest accuracy cost.

## Medaillon payoff on the HF backend

This is the headline insight: **Gold re-scoring from a persisted Silver table is the only way the HF backend is operationally viable**. Running Layers 1-3 takes ~75 seconds for 100 rows; re-scoring Gold from the same Silver takes <0.1 seconds (it's pure-Python risk / push-pull / velocity). For 1M rows with the HF backend, full re-runs are days, but a k-anonymity threshold change is milliseconds. Persist Silver to Delta; refresh Gold as often as your business rules change.

## Reproducing

```bash
pip install 'employee-voice-analytics[ml]'
python benchmarks/bench_hf.py --rows 100
```

The harness uses `subprocess.run` with its own timeout (no external
`timeout` binary needed; macOS doesn't ship one), so it works
identically on Linux and macOS. Each measurement writes one line to
`benchmarks/results_hf.csv` and a human report to
`benchmarks/RESULTS_hf.md`.
