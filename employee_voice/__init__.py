"""
Employee Voice Analytics — reusable sentiment / category / emotion /
theme / attrition-risk toolkit for HR feedback.

Public Python API — the recommended way to use this from code:

    from employee_voice import (
        analyze_feedback, analyze_file, analyze_spark,
        scrub, extract_aspects, score_risk, add_factors, add_risk_velocity,
        generate_synthetic,
    )

    # In-memory
    fact = analyze_feedback(my_df, redact=True, k_anonymity=5)

    # From a file
    artifacts = analyze_file("data/feedback.csv", "output/")

    # On Spark
    enriched = analyze_spark(spark_df, mlflow_experiment="/Shared/eva")

    # Per-layer building blocks
    redacted = scrub(df["Comment"], backend="presidio")
    aspects  = extract_aspects(["Great manager, awful salary", ...])
    factors  = detect_push_factors(comment)

    # Synthetic data for tests / demos
    df = generate_synthetic(n=250, seed=42)
"""
from __future__ import annotations

__version__ = "0.1.0"

# Re-export the public surface from employee_voice.api so that
# `from employee_voice import analyze_feedback` works at top level.
from .api import (  # noqa: F401
    analyze_feedback,
    analyze_file,
    analyze_spark,
    scrub,
    extract_aspects,
    detect_push_factors,
    detect_pull_factors,
    score_risk,
    add_factors,
    add_risk_velocity,
    extract_aspect_sentiment,
    extract_aspect_sentiment_dataframe,
    ASPECT_KEYWORDS,
    PUSH_FACTORS,
    PULL_FACTORS,
    PIIScrubber,
    KAnonymityConfig,
    apply_k_anonymity,
    clean_text,
    is_non_answer,
    generate_synthetic,
)

__all__ = [
    "__version__",
    "analyze_feedback",
    "analyze_file",
    "analyze_spark",
    "scrub",
    "extract_aspects",
    "detect_push_factors",
    "detect_pull_factors",
    "score_risk",
    "add_factors",
    "add_risk_velocity",
    "extract_aspect_sentiment",
    "extract_aspect_sentiment_dataframe",
    "ASPECT_KEYWORDS",
    "PUSH_FACTORS",
    "PULL_FACTORS",
    "PIIScrubber",
    "KAnonymityConfig",
    "apply_k_anonymity",
    "clean_text",
    "is_non_answer",
    "generate_synthetic",
]
