# Category model accuracy — single-machine benchmark

_Generated on arm64, Python 3.14.4, 100 rows._

## Caveat (read this first)

**This is NOT a real accuracy benchmark.** The synthetic data
has persona tags (Burnout, Exit, Happy, Neutral) but no per-row
HR category labels. We use the personas' *expected dominant*
category as a soft label: a Burnout comment is expected to be
labelled `Work-Life Balance` / `Workload` / `Compensation`, a
Happy comment is expected to be `Team and Collaboration` / `Culture`,
etc. Anything not matching the persona's expected list is a miss.

The *relative* numbers (BART vs distilbert vs SetFit) are
informative because the same heuristic applies to every
model. The *absolute* accuracy is not directly comparable to a
real labelled HR dataset.

## Results

| model | top-1 accuracy | rows scored | rows matched | elapsed sec |
|---|---:|---:|---:|---:|
| `bart-large-mnli` (zero-shot) | 60.0% | 60 | 36 | 49.4 |
| `distilbert-base-uncased-mnli` (zero-shot) | 48.3% | 60 | 29 | 14.6 |
| `all-MiniLM-L6-v2 (20-shot)` (few-shot) | 89.7% | 39 | 35 | 14.2 |

## How to read these

- If distilbert zero-shot is within 5-10% of BART zero-shot,
  the 5.9× speedup is essentially free. Use distilbert.
- If SetFit (few-shot distilbert) wins on accuracy, you get
  both the speedup AND better category quality than either
  zero-shot baseline. This is the production recommendation.
- If the SetFit run shows much worse numbers, that's likely
  the small training set (3 examples per persona) being too
  noisy. The next step is to label 50-100 real HR comments
  and re-run.
