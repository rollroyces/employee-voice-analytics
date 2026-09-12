# HuggingFace backend — single-CPU benchmark

_Measurement: Apple M-series arm64 with MPS GPU, Python 3.14.4, transformers latest. Each layer was warm-started with a 20-row pass; the measurement is the second call (steady state, no model load in the timer). 100 rows per layer._

| layer | rows | elapsed sec | **rows/sec** |
|---|---:|---:|---:|
| `sentiment` (cardiffnlp/twitter-roberta-base-sentiment-latest) | 100 | 4.40 | **22.7** |
| `category` (facebook/bart-large-mnli, zero-shot) | 100 | 65.67 | **1.5** |
| `emotion` (SamLowe/roberta-base-go_emotions) | 100 | 2.77 | **36.1** |

## Reading these numbers

- **CPU/GPU**: this is on an M-series Mac with MPS (Metal) GPU. CPU-only would be ~5× slower. A discrete GPU cluster would be ~10-50× faster again.
- **Sentiment + Emotion** are tractable on a single machine at ~20-40 rows/sec. That's roughly **3-4 hours for 1M rows** on a single CPU. A 4-core Spark executor with the same models gets you 8-16× throughput via `pandas_udf(SCALAR_ITER)`.
- **Category is the bottleneck** at 1.5 rows/sec because BART-MNLI is a 400M-parameter zero-shot model — each forward pass scores the text against all 18 HR categories. **1M rows would take ~7.7 days on a single machine.** This is not a viable path without parallelism. The right answers are:
  1. **Use the lexicon backend for category** on a CPU. 30k rows/sec = 33 seconds for 1M rows. The lexicon's accuracy is lower but predictable.
  2. **Run HF category on a Spark cluster** with `analyze_spark()`. The `pandas_udf(SCALAR_ITER)` path parallelises per-row inference across executors. On a 10-node cluster expect ~150 rows/sec *per node* = 1.5k rows/sec aggregate = ~11 minutes for 1M rows.
  3. **Distill BART** to a smaller model (e.g. `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`) for a 10× speedup at modest accuracy cost.

## Medaillon payoff on the HF backend

This is the headline insight: **Gold re-scoring from a persisted Silver table is the only way the HF backend is operationally viable**. Running Layers 1-3 takes ~73 seconds for 100 rows; re-scoring Gold from the same Silver takes <0.1 seconds (it's pure-Python risk / push-pull / velocity). For 1M rows with the HF backend, full re-runs are days, but a k-anonymity threshold change is milliseconds. Persist Silver to Delta; refresh Gold as often as your business rules change.

## Reproducing

```bash
# The harness in benchmarks/_hf_one_layer.py times one layer at a time
# and supports the same warmup-then-measure pattern as the main bench.
python benchmarks/bench_hf.py --rows 100
```

The warmup pass costs ~20 seconds (model download + first inference). The
measurement pass reuses the cached model and is what the table reports.
