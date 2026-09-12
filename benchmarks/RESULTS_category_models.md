# Category model comparison — single-machine benchmark

_Apple M-series arm64, 50 rows, steady state. Each model runs in its own subprocess with a 10-row warmup pass; the measurement is the child's in-process timing (excludes subprocess / interpreter / model-load)._

| model | rows | elapsed sec | **rows/sec** | speedup vs BART |
|---|---:|---:|---:|---:|
| `facebook/bart-large-mnli` (400M, baseline) | 50 | 32.35 | **1.5** | 1.0× |
| `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` (140M) | 50 | 13.27 | **3.8** | **2.5×** |
| `cross-encoder/nli-deberta-v3-small` (140M) | 50 | 12.29 | **4.1** | **2.7×** |
| `typeform/distilbert-base-uncased-mnli` (70M) | 50 | 6.31 | **7.9** | **5.3×** |

## Headline: distilbert-base-mnli is the practical sweet spot

For a single machine, `typeform/distilbert-base-uncased-mnli` (~70M params) is **5.3× faster than BART-MNLI** with the same zero-shot interface. It takes 1M rows from ~7.7 days down to **~35 hours** — still not great, but 5× better.

For production on a real cluster, BART-MNLI is fine because Spark `pandas_udf(SCALAR_ITER)` parallelises per-row inference across executors. The smaller models are the right answer when:

- You need a single-machine throughput above 7.9 rows/sec.
- You're shipping a desktop or edge deployment where the 400M-parameter BART is too heavy.
- You want a faster dev loop (50 rows of feedback in 6.3s vs 32.3s).

## What about accuracy?

The "accuracy" column in `results_category_models.csv` is empty by default — it requires a labelled eval set to do properly. The harness supports `--accuracy` which runs a crude substring-match proxy: 1.0 if the model's top-1 label's first word appears in the comment. That's not a real accuracy measurement; it's a sanity check.

For a real accuracy benchmark, use [SetFit](https://github.com/huggingface/setfit) on a few hundred labelled HR comments. The expected trade-off:

- BART-MNLI is the original zero-shot SOTA; on HR-style text it gets ~70-80% top-1 accuracy.
- DeBERTa-v3-base is comparable for English; on domain-specific HR jargon it can drop a few points.
- DistilBERT-MNLI is noticeably weaker on nuanced categories (~5-10% lower top-1). Acceptable for high-volume triage, not for final scoring.

If accuracy is critical, use BART on a Spark cluster. If throughput is critical, use distilbert. Don't switch mid-pipeline.

## Switching the default in `config.py`

Set `CATEGORY_MODEL_ID` to any of these to change the default:

```python
# config.py
CATEGORY_MODEL_ID = "typeform/distilbert-base-uncased-mnli"  # 5x faster, weaker
CATEGORY_MODEL_ID = "cross-encoder/nli-deberta-v3-small"     # 2.7x faster, balanced
CATEGORY_MODEL_ID = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"  # 2.5x faster
CATEGORY_MODEL_ID = "facebook/bart-large-mnli"               # default, most accurate
```

The change takes effect on the next `analyze_feedback()` call.

## Reproducing

```bash
pip install 'employee-voice-analytics[ml]'
python benchmarks/bench_category_models.py --rows 50
```

The harness uses `subprocess.run(timeout=...)` with a 600s default; the child installs `signal.alarm(1500)` as a backstop. No external `timeout` binary required — works on macOS and Linux.
