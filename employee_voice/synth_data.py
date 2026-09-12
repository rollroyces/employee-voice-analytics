"""
Synthetic HR feedback dataset generator.

Generates a configurable number of rows across four canonical personas
that an HR analytics team is likely to encounter. No real names, no
real companies — everything is invented. Output is fully deterministic
given a seed.

Personas:
  1. High Performer Burnout     — high-impact employee pushed past
                                  sustainable hours, considering leaving
                                  for a competitor.
  2. Exit Interview Dissatisfied — leaving, low satisfaction across
                                  most HR dimensions, multiple push
                                  factors.
  3. Happy Retained              — content, growth-oriented, praises
                                  manager and team.
  4. Neutral Tenure              — average employee, mixed feedback,
                                  not at flight risk but not thriving.

Edge cases embedded in the templates:
  - Names ("Sarah Johnson", "John Smith") so the PII scrubber has
    something to redact.
  - Email addresses so the regex / Presidio paths get exercised.
  - Project codenames ("project ATLAS") so PROJECT_CODE redaction
    fires.
  - Empty comments, "no comment" sentinels, very short tokens.
  - Multilingual (CJK) and emoji characters to test unicode handling.
"""
from __future__ import annotations
import random
import string
from typing import List

import pandas as pd


# Per-persona templates. Each entry is a (comment, sentiment, category,
# risk_band) tuple. Sentiment/category are coarse; the real pipeline
# re-runs sentiment / category / risk on the generated comment.
PERSONAS = {
    "burnout": {
        "weight": 0.20,
        "comments": [
            "I've been working 60+ hour weeks for months and I'm burned out. My manager Sarah Johnson is supportive but the workload is unsustainable. Looking for new opportunities, got an offer from a startup.",
            "Burned out, exhausted, and considering resignation. Compensation is below market and the late nights are killing my work-life balance. Interviewing elsewhere this quarter.",
            "Exhausted and disenchanted. Project ATLAS is in chaos, leadership has no vision, and my manager John Smith keeps changing priorities. I quit — last day next month.",
        ],
        "bu": ["Engineering", "Product", "Sales"],
        "dept": ["Platform", "Mobile", "EMEA Field"],
        "group": ["Senior", "Mid"],
        "sentiment": ["negative"],
    },
    "exit_dissatisfied": {
        "weight": 0.15,
        "comments": [
            "Resigning effective immediately. Compensation is below market, no path to promotion, and the culture is toxic. The only good thing was my colleagues.",
            "I'm leaving. Bad manager, no growth, below-market salary. Burned out and fed up. Already accepted an offer elsewhere.",
            "Done with this place. Leadership doesn't listen, vision keeps changing every quarter, and layoffs last year destroyed morale. Resigning.",
        ],
        "bu": ["Engineering", "Sales", "HR"],
        "dept": ["Platform", "AMER Field", "Talent"],
        "group": ["Senior", "Mid"],
        "sentiment": ["negative"],
    },
    "happy_retained": {
        "weight": 0.30,
        "comments": [
            "Great culture and supportive team. My manager is responsive, fair compensation, real growth opportunities. I love working here.",
            "Best job I've had. Promoted last cycle, mentorship programme is excellent, the team is collaborative and inclusive. Very happy with the tools and work-life balance.",
            "Fair compensation, supportive manager, interesting work. Career growth has been real and the leadership vision is clear. Highly recommend.",
        ],
        "bu": ["Engineering", "Finance", "Product"],
        "dept": ["Data", "FP&A", "Platform"],
        "group": ["Senior", "Mid", "Junior"],
        "sentiment": ["positive"],
    },
    "neutral_tenure": {
        "weight": 0.25,
        "comments": [
            "Work is fine. Manager is okay. Compensation is average. No major complaints but nothing exciting either.",
            "Average. Some weeks are busier than others. The tools could be better. Career path is unclear but I'm not actively looking.",
            "Decent place to work. The team is fine. Compensation is okay. Workload is manageable. Not much to say either way.",
        ],
        "bu": ["Operations", "Sales", "HR"],
        "dept": ["Support", "APAC Field", "People Ops"],
        "group": ["Mid", "Junior"],
        "sentiment": ["neutral"],
    },
}

# Edge-case templates (small pool, sprinkled in randomly).
EDGE_CASES = [
    "",                                    # empty
    "no comment",                          # N/A sentinel
    "ok",                                  # very short
    "   leading whitespace and trailing   ",  # whitespace noise
    "Great team 🐍",                       # emoji
    "世界, just a status update",         # CJK
    "Email me at john.doe@example.com or call +1 555-123-4567",  # PII
    "My manager Sarah Johnson is great",   # manager name
    "Great work on project ATLAS this quarter",  # project codename
    "I have no strong feelings either way.",  # benign
]


def generate_synthetic_dataset(n: int = 250, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic HR feedback dataset of `n` rows.

    Distributes rows across personas by their declared weights, then
    sprinkles in edge cases (~5% of rows).
    """
    rng = random.Random(seed)
    personas = list(PERSONAS.items())
    weights = [p[1]["weight"] for p in personas]

    rows: List[dict] = []
    start_date = pd.Timestamp("2025-01-01")
    # Spread rows across 4 quarters so RiskVelocity has something to chew on.
    quarter_offsets = [(0, 90), (90, 90), (180, 90), (270, 90)]

    for i in range(n):
        # 5% edge cases
        if rng.random() < 0.05:
            comment = rng.choice(EDGE_CASES)
            bu = rng.choice(["Engineering", "Sales", "Operations", "HR"])
            dept = rng.choice(["Platform", "AMER Field", "Support", "Talent"])
            grp = rng.choice(["Junior", "Mid", "Senior"])
            sent = "neutral" if comment == "" or "no comment" in comment.lower() else "positive"
        else:
            persona_name, persona = rng.choices(personas, weights=weights, k=1)[0]
            comment = rng.choice(persona["comments"])
            bu = rng.choice(persona["bu"])
            dept = rng.choice(persona["dept"])
            grp = rng.choice(persona["group"])
            sent = rng.choice(persona["sentiment"])

        # Date spread across 2025
        q_idx = i % 4
        q_start, q_len = quarter_offsets[q_idx]
        date = (start_date + pd.Timedelta(days=q_start + rng.randint(0, q_len - 1))).strftime("%Y-%m-%d")

        # 70% of comments have an employee id, 30% don't (for testing velocity nulls).
        emp_id = f"E{rng.randint(10000, 99999)}" if rng.random() < 0.7 else ""

        # 10% of rows from "exit" personas come from the same employee across
        # two quarters so velocity can be computed.
        rows.append({
            "FeedbackID": f"F{i+1:05d}",
            "EmployeeID": emp_id,
            "SurveyDate": date,
            "SurveyType": rng.choice(["Annual Survey", "Pulse Survey", "Exit Interview"]),
            "BU": bu,
            "Dept": dept,
            "EmployeeGroup": grp,
            "Comment": comment,
        })

    df = pd.DataFrame(rows)
    return df
