# Category model comparison — single-machine benchmark

_Generated on arm64, Python 3.14.4, 50 rows._

Category is the bottleneck in the HF backend (see `benchmarks/RESULTS_hf.md`). This bench measures whether a smaller zero-shot NLI model recovers most of the throughput without giving up too much accuracy.

| model | rows | elapsed sec | rows/sec | accuracy proxy |
|---|---:|---:|---:|---:|
| `bart-large-mnli (400M, baseline)` (facebook/bart-large-mnli) | 50 | 39.19 | 1.3 | n/a |
| `deberta-v3-base-mnli (140M)` (MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli) | 50 | 16.68 | 3.0 | n/a |
| `deberta-v3-small (140M)` (cross-encoder/nli-deberta-v3-small) | 50 | 13.79 | 3.6 | n/a |
| `distilbert-base-mnli (70M)` (typeform/distilbert-base-uncased-mnli) | 50 | 7.57 | 6.6 | n/a |

## Reading the accuracy proxy

The accuracy column is a *very* crude sanity check: it returns 1.0 if the top-1 model's first word appears as a substring of the comment. So for a comment like 'Compensation is below market', the model needs to predict any label starting with 'compensation' (e.g. 'Compensation and Benefits') to score 1.0. This is much weaker than a real label-quality benchmark on a held-out labelled set, but it's enough to spot a model that's completely lost. Use the [`setfit` library](https://github.com/huggingface/setfit) on a few hundred labelled comments for a real accuracy measurement.
