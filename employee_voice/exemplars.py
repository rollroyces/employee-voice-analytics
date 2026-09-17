"""
Few-shot exemplar pool for the LLM-backed category layer.

Why this exists
---------------
`prompts/category_v1.txt` has 3 hand-written few-shot examples. That
covers ~3 of the 18 HR categories in `config.CATEGORIES`, which
forces the LLM to guess for the rest. This module:

  1. Provides `EXEMPLARS`: a curated list of (comment, category,
     secondary) tuples covering all 18 categories. Built once,
     stable across runs.
  2. `format_exemplars()` — picks a balanced subset (default 18,
     one per category) and renders them as the few-shot block to
     inject into the prompt.
  3. `retrieve_exemplars()` — given an input comment, returns the
     K most similar exemplars from the pool (TF-IDF cosine sim).
     Used by `prompt_version="v2-retrieved"` mode to dynamically
     pick the most relevant few-shot examples per input.

Why static AND dynamic
----------------------
- Static exemplars (`category_v2`, 18 fixed) ensure every category
  is shown at least once in any prompt. Without this, retrieval
  alone can starve a category that just never matches a user
  comment (e.g. "Onboarding").
- Dynamic retrieval (`category_v2-retrieved`, top-K by similarity)
  picks the *most informative* examples for each input. The LLM
  learns the right shape faster when examples are close to the
  actual task.

Cost control
------------
At ~30 tokens per exemplar and 18 exemplars, the few-shot block
adds ~540 tokens to every prompt. At $0.15/M input tokens
(gpt-4o-mini), that's $0.00008 per call — negligible.
"""
from __future__ import annotations
import logging
import math
import re
from collections import Counter
from typing import Iterable, List, Optional, Sequence, Tuple

log = logging.getLogger(__name__)


# A curated pool of 50+ exemplars covering all 18 categories in
# `config.CATEGORIES`. Each entry: (comment, category, secondary_or_None).
EXEMPLARS: List[Tuple[str, str, Optional[str]]] = [
    # ---- Compensation and Benefits ----
    ("My base salary is way below market for my role and years of experience.",
     "Compensation and Benefits", None),
    ("The new health plan is confusing and the deductible is too high.",
     "Compensation and Benefits", None),
    ("Equity grants vest over 4 years but most of the value is in the last year, which feels risky.",
     "Compensation and Benefits", "Career Development"),

    # ---- Career Development ----
    ("There's no clear path from senior engineer to staff engineer here.",
     "Career Development", None),
    ("I'd love more budget for conference attendance and external training.",
     "Career Development", None),
    ("Got promoted last quarter after a tough year, the process felt fair.",
     "Career Development", "Recognition"),

    # ---- Leadership ----
    ("The VP's all-hands talks have stopped addressing the elephant in the room.",
     "Leadership", "Communication"),
    ("Senior leadership seems disconnected from what engineering actually does day-to-day.",
     "Leadership", "Culture and Values"),
    ("New CEO has set a clear direction for the next 18 months, refreshing.",
     "Leadership", None),

    # ---- Management ----
    ("My skip-level is the only one who gives me honest feedback.",
     "Management", None),
    ("Direct manager takes credit for team wins in leadership reviews.",
     "Management", "Culture and Values"),
    ("Manager is supportive during personal emergencies, very grateful.",
     "Management", "Recognition"),

    # ---- Work-Life Balance ----
    ("I can't take a real vacation because Slack follows me everywhere.",
     "Work-Life Balance", "Tools and Technology"),
    ("Remote work policy is generous and lets me be present for my kids.",
     "Work-Life Balance", None),
    ("On-call rotations bleed into weekends too often, even when nothing breaks.",
     "Work-Life Balance", "Workload"),

    # ---- Workload ----
    ("I'm on call every other weekend and the alert noise is brutal.",
     "Workload", "Work-Life Balance"),
    ("Sprint planning never accounts for production incidents, so we always slip.",
     "Workload", None),
    ("Three people left the team this quarter and we haven't backfilled, so the load doubled.",
     "Workload", "Management"),

    # ---- Culture and Values ----
    ("We talk about being customer-first but the engineering roadmap says otherwise.",
     "Culture and Values", "Company Strategy"),
    ("The values are on the wall but I've never seen them used in a real decision.",
     "Culture and Values", None),
    ("DEI efforts feel performative rather than structural.",
     "Culture and Values", "Diversity and Inclusion"),

    # ---- Team and Collaboration ----
    ("My team is the reason I stay — best coworkers I've worked with.",
     "Team and Collaboration", None),
    ("Cross-team handoffs are painful because we use three different ticketing systems.",
     "Team and Collaboration", "Tools and Technology"),
    ("Pair-programming sessions have made me a much better engineer.",
     "Team and Collaboration", "Career Development"),

    # ---- Tools and Technology ----
    ("The new laptop is great, but the build pipeline still takes 40 minutes.",
     "Tools and Technology", None),
    ("Internal admin tools are a decade behind what we ship to customers.",
     "Tools and Technology", "Customer or Stakeholder"),
    ("Please upgrade the CI runners, my morning standup is blocked on flaky tests.",
     "Tools and Technology", "Workload"),

    # ---- Physical Workspace ----
    ("Open-plan office is unbearable when the sales team is on calls all day.",
     "Physical Workspace", None),
    ("New HQ is gorgeous but the commute went from 20 minutes to 75.",
     "Physical Workspace", "Work-Life Balance"),
    ("The mother's room is always booked, which is awkward for new parents.",
     "Physical Workspace", "Diversity and Inclusion"),

    # ---- Communication ----
    ("Decisions get made in DMs that the rest of the team never hears about.",
     "Communication", "Culture and Values"),
    ("All-hands Q&A is too short and never gets to the hard questions.",
     "Communication", "Leadership"),
    ("Status updates are scattered across Slack, email, and a wiki nobody reads.",
     "Communication", "Tools and Technology"),

    # ---- Recognition ----
    ("Hard work goes unnoticed unless it's tied to a quarterly OKR.",
     "Recognition", "Performance Management"),
    ("My manager gave me a thoughtful shoutout in the team retro, made my week.",
     "Recognition", "Management"),
    ("Performance reviews feel like a checkbox exercise with no real feedback.",
     "Recognition", "Performance Management"),

    # ---- Job Satisfaction ----
    ("I genuinely enjoy the work but the politics are draining.",
     "Job Satisfaction", "Culture and Values"),
    ("Best job I've had in 10 years, even with the occasional crunch.",
     "Job Satisfaction", None),
    ("The mission doesn't inspire me the way it used to.",
     "Job Satisfaction", "Company Strategy"),

    # ---- Diversity and Inclusion ----
    ("Hiring panel had no women on it, again.",
     "Diversity and Inclusion", None),
    ("The ERG for LGBTQ+ employees has no executive sponsor.",
     "Diversity and Inclusion", "Leadership"),
    ("Promotion rates by demographic tell a clear story that nothing is changing.",
     "Diversity and Inclusion", "Performance Management"),

    # ---- Performance Management ----
    ("Perf review calibration meetings are opaque and political.",
     "Performance Management", None),
    ("360 feedback is taken seriously here, which is rare.",
     "Performance Management", "Culture and Values"),
    ("PIP process is adversarial rather than supportive.",
     "Performance Management", "Management"),

    # ---- Onboarding ----
    ("First month was chaos — no laptop for two weeks, no clear plan.",
     "Onboarding", "Tools and Technology"),
    ("Buddy system worked well, my onboarding buddy answered a thousand questions.",
     "Onboarding", "Team and Collaboration"),
    ("Onboarding docs are stale and contradict what engineering actually does.",
     "Onboarding", None),

    # ---- Company Strategy ----
    ("I don't understand how our AI investment makes money.",
     "Company Strategy", "Leadership"),
    ("The pivot last quarter was handled well, communication was clear.",
     "Company Strategy", "Communication"),
    ("Strategic priorities change every six months, hard to build momentum.",
     "Company Strategy", "Leadership"),

    # ---- Customer or Stakeholder ----
    ("Our biggest customer keeps changing requirements and we just say yes.",
     "Customer or Stakeholder", "Company Strategy"),
    ("Sales promises features that engineering hasn't agreed to deliver.",
     "Customer or Stakeholder", "Communication"),
    ("Customer support tickets sit in the backlog for months.",
     "Customer or Stakeholder", "Workload"),
]


def _tokenize(text: str) -> List[str]:
    """Simple whitespace + lowercase tokenisation. Drops punctuation."""
    return re.findall(r"[a-z0-9]+", text.lower())


def format_exemplars(
    exemplars: Optional[Sequence[Tuple[str, str, Optional[str]]]] = None,
    max_n: int = 18,
) -> str:
    """Render exemplars as a few-shot block for injection into the prompt.

    Picks up to `max_n` exemplars, balanced across categories. If the
    pool has more than `max_n` entries, we take one from each unique
    category first, then fill remaining slots with the most-frequent
    categories.

    Args:
      exemplars: explicit pool, defaults to the module-level EXEMPLARS.
      max_n: max exemplars to include. Default 18 = one per category.

    Returns:
      A string like:

        Examples (do NOT include these in the output):

          "salary is below market"
            -> {"category": "Compensation and Benefits"}

          ...

    Empty pool -> empty string (callers handle gracefully).
    """
    pool = list(exemplars if exemplars is not None else EXEMPLARS)
    if not pool:
        return ""

    # Group by category, preserve insertion order.
    by_cat: dict = {}
    for c, cat, sec in pool:
        by_cat.setdefault(cat, []).append((c, sec))

    cats = list(by_cat.keys())
    out: List[str] = ["Examples (do NOT include these in the output):\n"]
    n = 0
    # Round-robin one per category first.
    cat_iters = {cat: iter(rows) for cat, rows in by_cat.items()}
    while n < max_n and cat_iters:
        progressed = False
        for cat in list(cat_iters.keys()):
            if n >= max_n:
                break
            try:
                text, sec = next(cat_iters[cat])
            except StopIteration:
                del cat_iters[cat]
                continue
            example = f'  "{text}"\n    -> {{"category": "{cat}"'
            if sec:
                example += f', "secondary": "{sec}"'
            example += "}\n"
            out.append(example)
            n += 1
            progressed = True
        if not progressed:
            break

    return "".join(out)


def retrieve_exemplars(
    query: str,
    exemplars: Optional[Sequence[Tuple[str, str, Optional[str]]]] = None,
    k: int = 5,
) -> List[Tuple[str, str, Optional[str]]]:
    """Return the K exemplars most similar to `query`.

    Similarity is cosine over TF-IDF vectors built from the
    `comment` field. We use TF-IDF because the pool is small
    (<200 entries) and we don't need embeddings here.

    Falls back to round-robin if all similarities tie.
    """
    pool = list(exemplars if exemplars is not None else EXEMPLARS)
    if not pool or k <= 0:
        return []
    k = min(k, len(pool))

    # Build a tiny TF-IDF table.
    docs = [_tokenize(c) for c, _, _ in pool]
    q_tokens = _tokenize(query)
    if not q_tokens:
        # No tokens -> return the first K
        return pool[:k]

    # Document frequency
    df: Counter = Counter()
    for d in docs:
        for t in set(d):
            df[t] += 1
    n_docs = len(docs)

    def tfidf_vec(tokens: List[str]) -> dict:
        """Return a sparse dict {term: tfidf}."""
        tf: Counter = Counter(tokens)
        out = {}
        for t, c in tf.items():
            if t in df:
                idf = math.log((n_docs + 1) / (df[t] + 1)) + 1.0
                out[t] = c * idf
        return out

    def cosine(a: dict, b: dict) -> float:
        if not a or not b:
            return 0.0
        dot = sum(a.get(t, 0) * b.get(t, 0) for t in a)
        na = math.sqrt(sum(v * v for v in a.values()))
        nb = math.sqrt(sum(v * v for v in b.values()))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    qv = tfidf_vec(q_tokens)
    scored = [(cosine(qv, tfidf_vec(d)), i) for i, d in enumerate(docs)]
    # Sort by score desc, then by index asc for stability
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [pool[i] for _, i in scored[:k]]


def format_retrieved(query: str, k: int = 5,
                     pool: Optional[Sequence[Tuple[str, str, Optional[str]]]] = None) -> str:
    """Convenience: retrieve K exemplars for `query` and format them."""
    return format_exemplars(retrieve_exemplars(query, exemplars=pool, k=k))


__all__ = ["EXEMPLARS", "format_exemplars", "retrieve_exemplars", "format_retrieved"]
