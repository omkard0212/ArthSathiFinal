"""
ArthSathi — Full Recommendation Engine (Python)
================================================

Implements the exact algorithm specified in Section 4 of the product spec:
  Stage 1 — Three-valued eligibility filter (PASS / FAIL / UNKNOWN)
  Stage 2 — 8-feature weighted scoring
  Stage 3 — MMR diversity re-ranking + epsilon-greedy exploration
  Stage 4 — Per-user adaptive weight learning

See docs/ALGORITHM.md for the full mathematical documentation.

USAGE AS FASTAPI SERVICE
------------------------
  uvicorn recommendation_engine:app --host 127.0.0.1 --port 8001 --reload

ENDPOINTS
---------
  POST /recommend   { user_id, limit=10 }
  POST /event       { user_id, entity_type, entity_id, event_type, rank_position, session_id, client_event_id }
  POST /retrain     { scope: "user"|"global", user_id? }
  GET  /health
"""

from __future__ import annotations

import os
import sys
import io
import re
import json
import math
import time
import logging
import hashlib
import datetime
import traceback
from typing import Any, Optional, Literal

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import psycopg2
import psycopg2.extras
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

DB_CONFIG = {
    "host":     os.getenv("DB_HOST", "localhost"),
    "port":     int(os.getenv("DB_PORT", "5432")),
    "dbname":   os.getenv("DB_NAME", "arthsathi"),
    "user":     os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "postgres"),
}

MAX_RESULTS    = 10       # never return more than this
MMR_LAMBDA     = 0.8      # diversity trade-off (0=fully diverse, 1=fully ranked)
EPSILON_GREEDY = 0.10     # fraction of slots for exploration
MIN_WEIGHT     = 0.01     # floor for each weight
MAX_WEIGHT     = 0.60     # ceiling
LEARNING_RATE  = 0.10     # η₀ for per-user update
LR_DECAY_TAU   = 50       # η decays by 1/(1+t/τ)
GLOBAL_ALPHA   = 0.05     # regularization toward global prior

# INITIAL GLOBAL WEIGHTS (sum = 1.0)
# Documented in docs/ALGORITHM.md
GLOBAL_WEIGHTS = {
    "f1_goal_relevance":    0.25,
    "f2_financial_benefit": 0.20,
    "f3_urgency_match":     0.15,
    "f4_eligibility_margin": 0.12,
    "f5_ease_of_access":    0.08,
    "f6_coverage_priority": 0.08,
    "f7_category_affinity": 0.07,
    "f8_recency_deadline":  0.05,
}

# Reward values per event type (for adaptive learning)
EVENT_REWARDS = {
    "apply":    1.0,
    "save":     0.6,
    "click":    0.3,
    "listen":   0.15,
    "dismiss": -0.5,
    "impression": -0.05,
}

log = logging.getLogger("engine")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# ─────────────────────────────────────────────────────────────────────────────
# Database helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_db():
    return psycopg2.connect(**DB_CONFIG, cursor_factory=psycopg2.extras.RealDictCursor)


def get_user_weights(conn, user_id: str) -> dict:
    """
    Load per-user weights from user_weight_profiles.
    Falls back to global prior if no record exists.
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT * FROM user_weight_profiles WHERE user_id = %s",
            (user_id,)
        )
        row = cur.fetchone()

    if row:
        return {
            "f1_goal_relevance":    float(row["w_goal_relevance"]),
            "f2_financial_benefit": float(row["w_financial_benefit"]),
            "f3_urgency_match":     float(row["w_urgency_match"]),
            "f4_eligibility_margin": float(row["w_eligibility_margin"]),
            "f5_ease_of_access":    float(row["w_ease_of_access"]),
            "f6_coverage_priority": float(row["w_coverage_priority"]),
            "f7_category_affinity": float(row["w_category_affinity"]),
            "f8_recency_deadline":  float(row["w_recency_deadline"]),
            "_interaction_count":   int(row["interaction_count"]),
            "_category_affinity":   row["category_affinity"] or {},
        }

    # Cold start — use global prior
    with conn.cursor() as cur:
        cur.execute(
            "SELECT * FROM global_weight_profile WHERE is_active = TRUE ORDER BY version DESC LIMIT 1"
        )
        g = cur.fetchone()

    w = dict(GLOBAL_WEIGHTS)
    if g:
        w["f1_goal_relevance"]    = float(g["w_goal_relevance"])
        w["f2_financial_benefit"] = float(g["w_financial_benefit"])
        w["f3_urgency_match"]     = float(g["w_urgency_match"])
        w["f4_eligibility_margin"] = float(g["w_eligibility_margin"])
        w["f5_ease_of_access"]    = float(g["w_ease_of_access"])
        w["f6_coverage_priority"] = float(g["w_coverage_priority"])
        w["f7_category_affinity"] = float(g["w_category_affinity"])
        w["f8_recency_deadline"]  = float(g["w_recency_deadline"])
    w["_interaction_count"] = 0
    w["_category_affinity"] = {}
    return w


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: Three-valued eligibility filter
# ─────────────────────────────────────────────────────────────────────────────

EligibilityResult = Literal["eligible", "ineligible", "likely", "unknown"]

def check_criterion(user_val: Any, scheme_val: Any, check_type: str) -> str:
    """
    Check a single eligibility criterion.
    Returns 'pass', 'fail', or 'unknown'.

    check_type: 'min', 'max', 'array_in', 'bool_required', 'bool_forbidden'
    """
    # Unknown if scheme has no restriction on this criterion
    if scheme_val is None or scheme_val == [] or scheme_val == "":
        return "pass"

    # Unknown if user hasn't answered this field
    if user_val is None or user_val == "":
        return "unknown"

    if check_type == "min":
        try:
            return "pass" if float(user_val) >= float(scheme_val) else "fail"
        except (ValueError, TypeError):
            return "unknown"

    if check_type == "max":
        try:
            return "pass" if float(user_val) <= float(scheme_val) else "fail"
        except (ValueError, TypeError):
            return "unknown"

    if check_type == "array_in":
        # scheme_val is a list; user_val must be in it
        if isinstance(scheme_val, (list, tuple)):
            return "pass" if str(user_val) in [str(v) for v in scheme_val] else "fail"
        return "unknown"

    if check_type == "bool_required":
        # Scheme requires this; user must have it
        if not scheme_val:
            return "pass"
        return "pass" if bool(user_val) else "fail"

    return "unknown"


def evaluate_eligibility(profile: dict, scheme: dict) -> tuple[EligibilityResult, list, int]:
    """
    Three-valued eligibility evaluation.
    
    Returns:
      (status, missing_criteria, unknown_count)
      status: 'eligible' | 'likely' | 'ineligible' | 'unknown'
      missing_criteria: list of field names that are UNKNOWN (user hasn't answered)
      unknown_count: total unknown criterion count (for penalty)
    """
    criteria_checks = []
    missing = []

    def chk(name, user_val, scheme_val, ctype):
        result = check_criterion(user_val, scheme_val, ctype)
        criteria_checks.append((name, result))
        if result == "unknown":
            missing.append(name)
        return result

    # Age
    chk("min_age",    profile.get("age"), scheme.get("min_age"), "min")
    chk("max_age",    profile.get("age"), scheme.get("max_age"), "max")

    # Income
    income = profile.get("annual_income") or (
        (profile.get("monthly_income") or 0) * 12
    )
    chk("min_income", income, scheme.get("min_annual_income"), "min")
    chk("max_income", income, scheme.get("max_annual_income"), "max")

    # State
    s_states = scheme.get("applicable_states") or scheme.get("eligible_states_exact")
    if s_states and isinstance(s_states, (list, tuple)) and len(s_states) > 0:
        # Filter out "All" which means all-India
        real_states = [s for s in s_states if s and str(s).strip().lower() not in ("all", "all india", "")]
        if real_states:
            chk("state", profile.get("state"), real_states, "array_in")
    # (NULL applicable_states = all-India, always pass)

    # Gender
    chk("gender", profile.get("gender"), scheme.get("eligible_genders"), "array_in")

    # Caste
    chk("caste", profile.get("caste_category"), scheme.get("eligible_castes"), "array_in")

    # Occupation
    chk("occupation", profile.get("occupation"), scheme.get("eligible_occupations"), "array_in")

    # Ration card
    chk("ration_card", profile.get("ration_card_type"), scheme.get("eligible_ration_cards"), "array_in")

    # Residence
    chk("residence", profile.get("residence_type"), scheme.get("eligible_residence"), "array_in")

    # Boolean requirements
    chk("bank_account", profile.get("has_bank_account") or profile.get("has_jan_dhan_account"),
        scheme.get("requires_bank_account"), "bool_required")
    chk("aadhaar", profile.get("has_aadhaar"),
        scheme.get("requires_aadhaar"), "bool_required")
    chk("bpl", profile.get("is_bpl") or (
        profile.get("ration_card_type") in ("aay", "phh")),
        scheme.get("requires_bpl"), "bool_required")
    chk("disability", profile.get("has_disability"),
        scheme.get("requires_disability"), "bool_required")
    chk("widow", profile.get("is_widow"),
        scheme.get("requires_widow"), "bool_required")
    chk("student", profile.get("is_student"),
        scheme.get("requires_student"), "bool_required")
    chk("farmer", (profile.get("occupation") in ("farmer", "marginal_farmer")),
        scheme.get("requires_farmer"), "bool_required")

    # Count outcomes
    fails    = sum(1 for _, r in criteria_checks if r == "fail")
    unknowns = sum(1 for _, r in criteria_checks if r == "unknown")

    if fails > 0:
        return "ineligible", missing, unknowns

    if unknowns == 0:
        return "eligible", missing, 0

    return "likely", missing, unknowns


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: 8-feature scoring
# ─────────────────────────────────────────────────────────────────────────────

def f1_goal_relevance(profile: dict, scheme: dict) -> float:
    """
    Feature 1: Goal relevance using ALL profile fields for maximum accuracy.
    Uses: financial_goal, goal_description, occupation, caste, gender,
    disability, widow, student, minority, sub_occupation, residence.
    """
    goal    = (profile.get("financial_goal") or "").lower()
    desc    = (profile.get("goal_description") or "").lower()
    occ     = (profile.get("occupation") or "").lower().replace("_", " ")
    sub_occ = (profile.get("sub_occupation") or "").lower()
    caste   = (profile.get("caste_category") or "").lower()
    gender  = (profile.get("gender") or "").lower()
    marital = (profile.get("marital_status") or "").lower()
    religion = (profile.get("religion") or "").lower()
    is_disabled = bool(profile.get("has_disability"))
    is_student  = bool(profile.get("is_student")) or profile.get("occupation") == "student"
    is_widow    = bool(profile.get("is_widow")) or marital == "widowed"
    has_girl    = bool(profile.get("has_girl_child"))
    is_minority = bool(profile.get("is_minority"))

    # Build rich profile token list from all relevant fields
    profile_tokens = list(filter(None, goal.split() + desc.split() + occ.split() + sub_occ.split()))
    if caste in ("sc", "st", "obc", "ews"):
        profile_tokens.append(caste)
    if is_disabled:
        profile_tokens += ["disability", "disabled", "divyang", "pwd"]
    if is_student:
        profile_tokens += ["student", "scholarship", "education", "fellowship"]
    if is_widow:
        profile_tokens += ["widow", "vidhwa"]
    if has_girl:
        profile_tokens += ["girl", "daughter", "beti", "mahila"]
    if is_minority:
        profile_tokens += ["minority", religion]
    if profile.get("is_senior_citizen"):
        profile_tokens += ["senior", "pension", "old age"]
    if "fisherman" in occ:
        profile_tokens += ["fisher", "marine", "fishing"]
    if "weaver" in occ:
        profile_tokens += ["weaver", "handloom", "textile"]
    if "artisan" in occ:
        profile_tokens += ["artisan", "handicraft", "craft", "khadi"]

    if not profile_tokens:
        return 0.3

    # Scheme tokens (name + tags + category + eligibility text)
    scheme_name   = (scheme.get("name") or {}).get("en", "").lower()
    scheme_tags   = [t.lower() for t in (scheme.get("tags") or [])]
    scheme_cat    = (scheme.get("category") or "").lower()
    scheme_subcat = (scheme.get("sub_category") or "").lower()
    elig_text     = ((scheme.get("eligibility_text") or {}).get("en") or "").lower()[:300]
    scheme_tokens = set((
        scheme_name + " " + " ".join(scheme_tags) + " " +
        scheme_cat + " " + scheme_subcat + " " + elig_text
    ).split())

    stop = {"the", "for", "and", "are", "has", "was", "with", "this", "that", "have", "from", "not", "can"}
    profile_tokens = [w for w in profile_tokens if len(w) >= 3 and w not in stop]
    if not profile_tokens:
        return 0.3

    matches = sum(1 for w in profile_tokens if any(t in w or w in t for t in scheme_tokens))
    overlap = matches / len(profile_tokens)

    # Direct goal -> category match (strongest signal)
    GOAL_CAT = {
        "farming":            ["agriculture", "farmer", "kisan", "crop", "seed", "irrigation", "fertilizer", "horticulture"],
        "housing":            ["housing", "house", "awas", "shelter", "pucca", "construction", "habitat"],
        "health":             ["health", "medical", "hospital", "ayushman", "arogya", "sanitation", "nutrition", "medicine"],
        "education":           ["education", "scholarship", "school", "college", "student", "fellowship", "training", "post-matric"],
        "employment":          ["employment", "job", "wage", "rozgar", "livelihood", "nrega", "labour", "labor"],
        "insurance":           ["insurance", "bima", "coverage", "premium", "accident", "life"],
        "pension":             ["pension", "old age", "senior", "retirement", "apy", "nps"],
        "business":            ["business", "enterprise", "mudra", "msme", "startup", "self-employ", "trade", "entrepreneur"],
        "daughter":            ["girl", "daughter", "beti", "mahila", "women", "sukanya", "marriage"],
        "loan":                ["loan", "credit", "mudra", "kcc", "concessional", "interest", "finance"],
        "savings":             ["savings", "deposit", "nsc", "post office"],
        "disability_support":  ["disability", "divyang", "pwd", "handicap", "rehabilitation"],
        "water_sanitation":    ["sanitation", "toilet", "water", "swachh", "latrine"],
        "skill_training":      ["skill", "training", "vocational", "pmkvy", "iti", "craft", "apprentice"],
    }
    cat_kws = GOAL_CAT.get(goal, [])
    scheme_text = " ".join(scheme_tokens)
    direct_hit = 0.35 if any(kw in scheme_text for kw in cat_kws) else 0

    # Targeted identity boosts
    boost = 0.0
    if caste in ("sc", "st") and any(w in scheme_tokens for w in [caste, "scheduled", "dalit", "tribal"]):
        boost += 0.15
    if caste == "obc" and any(w in scheme_tokens for w in ["obc", "backward"]):
        boost += 0.10
    if is_disabled and any(w in scheme_tokens for w in ["disab", "divyang", "pwd", "rehabilitation"]):
        boost += 0.20
    if gender == "female" and any(w in scheme_tokens for w in ["women", "mahila", "female", "girl", "stree"]):
        boost += 0.10
    if is_widow and any(w in scheme_tokens for w in ["widow", "vidhwa"]):
        boost += 0.15
    if occ in ("farmer", "marginal farmer") and any(w in scheme_tokens for w in ["farmer", "kisan", "agriculture"]):
        boost += 0.15
    if is_student and any(w in scheme_tokens for w in ["student", "scholar", "fellowship", "matric"]):
        boost += 0.15
    if is_minority and any(w in scheme_tokens for w in ["minority", religion, "nmdfc"]):
        boost += 0.15
    if "fisherman" in occ and any(w in scheme_tokens for w in ["fisher", "fish", "marine"]):
        boost += 0.20
    if any(w in occ for w in ["weaver", "artisan"]) and any(w in scheme_tokens for w in ["artisan", "weaver", "handicraft", "handloom", "khadi"]):
        boost += 0.20

    return min(1.0, round(overlap + direct_hit + boost, 4))


def f2_financial_benefit(profile: dict, scheme: dict) -> float:
    """
    Feature 2: Benefit magnitude relative to user's income and goal.

    - cash_direct / loan / subsidy / scholarship schemes score by amount
    - service / training schemes get a baseline 0.4
    - insurance schemes scored by coverage relative to income
    """
    annual_income = float(profile.get("annual_income") or
                          (profile.get("monthly_income") or 0) * 12 or 1)
    goal_amount   = float(profile.get("goal_amount") or 0)

    ben_min = float(scheme.get("benefit_amount_min") or 0)
    ben_max = float(scheme.get("benefit_amount_max") or ben_min)
    benefit = (ben_min + ben_max) / 2 if ben_min > 0 else 0

    ben_type = scheme.get("benefit_type") or "other"

    if benefit <= 0:
        # Non-monetary: service, training, equipment
        if ben_type in ("service", "training", "equipment"):
            return 0.4
        return 0.25

    # Goal amount coverage
    goal_coverage = min(1.0, benefit / goal_amount) if goal_amount > 0 else 0.5

    # Income significance: log-scaled ratio
    ratio = benefit / annual_income
    if ratio >= 1.0:    income_sig = 1.0
    elif ratio >= 0.5:  income_sig = 0.85
    elif ratio >= 0.2:  income_sig = 0.70
    elif ratio >= 0.1:  income_sig = 0.60
    elif ratio >= 0.05: income_sig = 0.50
    elif ratio >= 0.01: income_sig = 0.35
    else:               income_sig = 0.20

    # Blend goal coverage and income significance
    score = 0.5 * goal_coverage + 0.5 * income_sig

    # Boost for cash/direct transfer (most useful for immediate needs)
    if ben_type == "cash_direct":
        score = min(1.0, score * 1.15)

    return round(score, 4)


def f3_urgency_match(profile: dict, scheme: dict) -> float:
    """
    Feature 3: How well does the scheme's delivery speed match the user's urgency?

    urgency_level: 1=no hurry → 5=emergency
    Fast schemes (one-time cash, insurance) score high for urgent users.
    Long-term schemes (pension, housing) score high for non-urgent.
    """
    urgency = int(profile.get("urgency_level") or 3)
    urg_norm = urgency / 5.0  # 0.2 – 1.0

    ben_type  = scheme.get("benefit_type") or "other"
    frequency = scheme.get("benefit_frequency") or ""
    deadline_str = scheme.get("application_deadline")

    # Speed proxy by benefit type
    SPEED = {
        "cash_direct": 0.9,   # DBT transfers are fast
        "insurance":   0.8,   # Cover available immediately after enrolment
        "scholarship": 0.7,   # Delayed but single disbursement
        "loan":        0.7,
        "subsidy":     0.6,
        "service":     0.5,
        "housing":     0.3,   # Construction takes time
        "pension":     0.2,   # Long-term
        "training":    0.3,
        "other":       0.5,
    }
    speed = SPEED.get(ben_type, 0.5)

    # High urgency → fast schemes score better
    urgency_speed_match = 1 - abs(urg_norm - speed)

    # Deadline proximity boost (closing soon = act now)
    deadline_boost = 0
    if deadline_str:
        try:
            deadline = datetime.date.fromisoformat(str(deadline_str)[:10])
            days_left = (deadline - datetime.date.today()).days
            if 0 < days_left <= 30:
                deadline_boost = 0.15  # closing very soon
            elif 0 < days_left <= 90:
                deadline_boost = 0.08
        except Exception:
            pass

    return min(1.0, urgency_speed_match + deadline_boost)


def f4_eligibility_margin(profile: dict, scheme: dict,
                           eligibility_status: str, unknown_count: int) -> float:
    """
    Feature 4: Comfort margin within the eligibility window.
    "Likely" schemes get penalized by unknown_count.

    Components:
    - Age margin: how far is the user from the nearest age boundary?
    - Income margin: how far below the ceiling?
    - Status penalty for 'likely' (missing info)
    """
    scores = []

    # Age margin
    age      = profile.get("age")
    min_age  = scheme.get("min_age")
    max_age  = scheme.get("max_age")
    if age and (min_age or max_age):
        lo = min_age or 0
        hi = max_age or 120
        age_range = hi - lo
        if age_range > 0:
            dist = min(age - lo, hi - age)
            age_score = max(0.1, min(1.0, dist / (age_range / 2)))
            scores.append(age_score)

    # Income margin
    income = float(profile.get("annual_income") or
                   (profile.get("monthly_income") or 0) * 12 or 0)
    max_inc = float(scheme.get("max_annual_income") or 0)
    if income > 0 and max_inc > 0:
        # Closer to ceiling = lower margin
        ratio = income / max_inc
        income_score = max(0.1, 1.0 - ratio * 0.8)
        scores.append(income_score)

    base = sum(scores) / len(scores) if scores else 0.6

    # Penalize for unknown criteria
    unknown_penalty = unknown_count * 0.08
    if eligibility_status == "likely":
        base -= unknown_penalty

    return max(0.05, min(1.0, base))


def f5_ease_of_access(profile: dict, scheme: dict) -> float:
    """
    Feature 5: How accessible is this scheme for a rural, low-income user?

    Factors:
    - Fewer documents required = easier
    - User already has required documents (bank account, Aadhaar, ration card)
    - Online/CSC application available
    - No collateral required
    """
    score = 0.5  # base

    # Documents required
    docs = scheme.get("documents_required") or []
    if isinstance(docs, (list, str)):
        doc_count = len(docs) if isinstance(docs, list) else len(str(docs).split(","))
        if doc_count <= 2:     score += 0.15
        elif doc_count <= 4:   score += 0.05
        elif doc_count >= 8:   score -= 0.10

    # User has required docs
    has_bank    = profile.get("has_bank_account") or profile.get("has_jan_dhan_account")
    has_aadhaar = profile.get("has_aadhaar")
    req_bank    = scheme.get("requires_bank_account")
    req_aadhaar = scheme.get("requires_aadhaar")

    if req_bank and has_bank:    score += 0.10
    if req_aadhaar and has_aadhaar: score += 0.10
    if req_bank and not has_bank:   score -= 0.15  # barrier
    if req_aadhaar and not has_aadhaar: score -= 0.10

    # Jan Dhan = designed for poor = easy
    if profile.get("has_jan_dhan_account"):
        score += 0.05

    # Application mode (check in app_modes or application_url)
    app_url = scheme.get("application_url") or ""
    if "online" in app_url.lower() or "myscheme" in app_url.lower():
        score += 0.05

    return max(0.0, min(1.0, score))


def f6_coverage_priority(profile: dict, scheme: dict) -> float:
    """
    Feature 6: Risk coverage priority.
    Boosts insurance/health/disability schemes for vulnerable users
    who currently lack coverage.
    """
    ben_type = scheme.get("benefit_type") or "other"
    category  = scheme.get("category") or "other"

    # Non-insurance schemes get base score
    if ben_type != "insurance" and category not in ("health", "insurance", "disability"):
        return 0.3

    score = 0.4  # base for insurance/health schemes

    # Higher priority if user is uninsured and high-risk
    dependents  = int(profile.get("num_dependents") or 0)
    has_dis     = bool(profile.get("has_disability"))
    income      = float(profile.get("annual_income") or
                        (profile.get("monthly_income") or 0) * 12 or 0)

    if dependents > 2:    score += 0.15
    if has_dis:           score += 0.10
    if income < 100_000:  score += 0.10  # very low income = higher priority
    if income < 50_000:   score += 0.10

    return min(1.0, score)


def f7_category_affinity(profile_weights: dict, scheme: dict) -> float:
    """
    Feature 7: Learned per-user affinity for this scheme's category.
    Starts at 0.5 (neutral) for all users; updated from interaction events.
    """
    category = scheme.get("category") or "other"
    affinities = profile_weights.get("_category_affinity") or {}
    return float(affinities.get(category, 0.5))


def f8_recency_deadline(scheme: dict) -> float:
    """
    Feature 8: Freshness and deadline urgency.
    - Newly scraped = possibly newly launched
    - Closing soon = should surface first
    """
    score = 0.5

    # Recent scrape = recently added/updated
    last_scraped = scheme.get("last_scraped_at")
    if last_scraped:
        try:
            if isinstance(last_scraped, str):
                scraped = datetime.datetime.fromisoformat(last_scraped.replace("Z", "+00:00"))
            else:
                scraped = last_scraped
            if scraped.tzinfo:
                now = datetime.datetime.now(datetime.timezone.utc)
            else:
                now = datetime.datetime.now()
            days_old = (now - scraped).days
            if days_old <= 7:    score += 0.10
            elif days_old <= 30: score += 0.05
        except Exception:
            pass

    # Deadline proximity
    deadline_str = scheme.get("application_deadline")
    if deadline_str:
        try:
            deadline = datetime.date.fromisoformat(str(deadline_str)[:10])
            days_left = (deadline - datetime.date.today()).days
            if 0 < days_left <= 14:  score += 0.25
            elif 0 < days_left <= 30: score += 0.15
            elif 0 < days_left <= 90: score += 0.08
            elif days_left < 0:       score = 0.0   # expired
        except Exception:
            pass

    return min(1.0, max(0.0, score))


def compute_score(profile: dict, scheme: dict, weights: dict,
                  eligibility_status: str, unknown_count: int) -> tuple[float, dict]:
    """
    Compute the composite score and per-feature breakdown.

    score = Σ w_k * f_k − unknown_penalty
    Returns (score, breakdown_dict)
    """
    f1 = f1_goal_relevance(profile, scheme)
    f2 = f2_financial_benefit(profile, scheme)
    f3 = f3_urgency_match(profile, scheme)
    f4 = f4_eligibility_margin(profile, scheme, eligibility_status, unknown_count)
    f5 = f5_ease_of_access(profile, scheme)
    f6 = f6_coverage_priority(profile, scheme)
    f7 = f7_category_affinity(weights, scheme)
    f8 = f8_recency_deadline(scheme)

    breakdown = {
        "f1_goal_relevance":    round(f1, 4),
        "f2_financial_benefit": round(f2, 4),
        "f3_urgency_match":     round(f3, 4),
        "f4_eligibility_margin": round(f4, 4),
        "f5_ease_of_access":    round(f5, 4),
        "f6_coverage_priority": round(f6, 4),
        "f7_category_affinity": round(f7, 4),
        "f8_recency_deadline":  round(f8, 4),
    }

    # Weight keys match feature names
    w_map = {
        "f1": weights.get("f1_goal_relevance", GLOBAL_WEIGHTS["f1_goal_relevance"]),
        "f2": weights.get("f2_financial_benefit", GLOBAL_WEIGHTS["f2_financial_benefit"]),
        "f3": weights.get("f3_urgency_match", GLOBAL_WEIGHTS["f3_urgency_match"]),
        "f4": weights.get("f4_eligibility_margin", GLOBAL_WEIGHTS["f4_eligibility_margin"]),
        "f5": weights.get("f5_ease_of_access", GLOBAL_WEIGHTS["f5_ease_of_access"]),
        "f6": weights.get("f6_coverage_priority", GLOBAL_WEIGHTS["f6_coverage_priority"]),
        "f7": weights.get("f7_category_affinity", GLOBAL_WEIGHTS["f7_category_affinity"]),
        "f8": weights.get("f8_recency_deadline", GLOBAL_WEIGHTS["f8_recency_deadline"]),
    }

    score = (
        w_map["f1"] * f1 + w_map["f2"] * f2 + w_map["f3"] * f3 +
        w_map["f4"] * f4 + w_map["f5"] * f5 + w_map["f6"] * f6 +
        w_map["f7"] * f7 + w_map["f8"] * f8
    )

    # Penalty for 'likely' (missing info)
    if eligibility_status == "likely":
        penalty = unknown_count * 0.04
        score -= penalty

    return round(max(0.0, min(1.0, score)), 5), breakdown


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3: MMR Diversity Re-ranking
# ─────────────────────────────────────────────────────────────────────────────

def scheme_similarity(s1: dict, s2: dict) -> float:
    """
    Cosine-like similarity between two schemes based on category + tags.
    Returns 0.0–1.0.
    """
    cat1 = s1.get("category") or ""
    cat2 = s2.get("category") or ""
    tags1 = set(s1.get("tags") or [])
    tags2 = set(s2.get("tags") or [])

    cat_sim  = 1.0 if cat1 == cat2 else 0.0
    tag_sim  = len(tags1 & tags2) / max(len(tags1 | tags2), 1)

    return 0.6 * cat_sim + 0.4 * tag_sim


def mmr_rerank(candidates: list, limit: int) -> list:
    """
    Maximal Marginal Relevance re-ranking.

    Greedily picks the next item that maximises:
      λ * score(s) − (1−λ) * max_sim(s, already_selected)

    This prevents the top 10 from being dominated by very similar schemes.
    Also reserves ~EPSILON_GREEDY * limit slots for exploration.
    """
    if len(candidates) <= limit:
        return candidates

    selected   = []
    remaining  = list(candidates)
    explore_k  = max(1, round(EPSILON_GREEDY * limit))
    regular_k  = limit - explore_k

    # Regular slots: MMR greedy selection
    for _ in range(regular_k):
        if not remaining:
            break

        if not selected:
            # First pick: highest score
            best = max(remaining, key=lambda x: x["score"])
        else:
            # MMR criterion
            def mmr_score(s):
                max_sim = max(scheme_similarity(s, sel) for sel in selected)
                return MMR_LAMBDA * s["score"] - (1 - MMR_LAMBDA) * max_sim
            best = max(remaining, key=mmr_score)

        selected.append(best)
        remaining.remove(best)

    # Exploration slot(s): pick from categories not yet represented
    selected_cats = {s.get("category") for s in selected}
    novel = [s for s in remaining if s.get("category") not in selected_cats]
    if novel:
        # Pick the highest-scoring novel scheme
        for s in sorted(novel, key=lambda x: x["score"], reverse=True)[:explore_k]:
            s["_is_exploration"] = True
            selected.append(s)
            explore_k -= 1
            if explore_k <= 0:
                break

    # Fill remaining slots with highest scoring remaining
    while explore_k > 0 and remaining:
        nxt = max(remaining, key=lambda x: x["score"])
        nxt["_is_exploration"] = True
        selected.append(nxt)
        remaining.remove(nxt)
        explore_k -= 1

    return selected


# ─────────────────────────────────────────────────────────────────────────────
# Explain function
# ─────────────────────────────────────────────────────────────────────────────

def explain(profile: dict, scheme: dict, breakdown: dict,
            eligibility_status: str) -> str:
    """
    Generate a plain-language explanation for why this scheme was recommended.
    Used for both display and TTS.

    Example: "You qualify because you are a woman farmer in Maharashtra with
    income under Rs.1 lakh. This gives Rs.6,000 per year directly to your
    bank account."
    """
    reasons = []
    name = (scheme.get("name") or {}).get("en", "this scheme")

    # Top scoring features
    sorted_features = sorted(breakdown.items(), key=lambda x: x[1], reverse=True)

    for feat_name, feat_val in sorted_features[:3]:
        if feat_val < 0.3:
            continue
        if "goal" in feat_name and feat_val >= 0.5:
            goal = profile.get("financial_goal") or "your goal"
            reasons.append(f"matches your need ({goal})")
        elif "financial" in feat_name and feat_val >= 0.5:
            ben_max = scheme.get("benefit_amount_max")
            if ben_max:
                reasons.append(f"provides Rs.{int(ben_max):,} benefit")
        elif "eligibility" in feat_name and feat_val >= 0.5:
            reasons.append("you comfortably meet the eligibility criteria")
        elif "access" in feat_name and feat_val >= 0.6:
            reasons.append("easy to apply with your existing documents")
        elif "urgency" in feat_name and feat_val >= 0.5:
            reasons.append("benefit arrives quickly for your urgent need")

    # Add caste/gender match if applicable
    user_caste = profile.get("caste_category")
    scheme_castes = scheme.get("eligible_castes") or []
    if user_caste and scheme_castes and user_caste in scheme_castes:
        reasons.append(f"specifically for {user_caste.upper()} category")

    user_gender = profile.get("gender")
    scheme_genders = scheme.get("eligible_genders") or []
    if user_gender and scheme_genders and user_gender in scheme_genders:
        reasons.append(f"designed for {user_gender} beneficiaries")

    if eligibility_status == "likely":
        reasons.append("answer a few more questions to confirm eligibility")

    if reasons:
        return f"Recommended because: " + "; ".join(reasons[:3]) + "."
    return f"This scheme may be relevant to your profile."


# ─────────────────────────────────────────────────────────────────────────────
# Main recommendation pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_recommendations(profile: dict, items: list, weights: dict,
                        item_type: str = "scheme",
                        limit: int = MAX_RESULTS) -> list:
    """
    Full pipeline: eligibility filter → score → MMR re-rank.

    profile — combined user + financial_profile dict
    items   — list of scheme or insurance_policy rows (from DB)
    weights — user's weight vector
    item_type — 'scheme' | 'insurance'
    limit   — max results to return

    Returns list of result dicts with score, breakdown, explain_text.
    """
    scored = []

    for item in items:
        # Stage 1: Three-valued eligibility
        status, missing, unknown_count = evaluate_eligibility(profile, item)

        if status == "ineligible":
            continue  # Hard fail — never show

        # Stage 2: Score
        score, breakdown = compute_score(
            profile, item, weights, status, unknown_count
        )

        # Build result
        result = dict(item)
        result["score"]            = score
        result["score_breakdown"]  = breakdown
        result["eligibility_status"] = status
        result["missing_criteria"] = missing
        result["explain_text"]     = explain(profile, item, breakdown, status)

        scored.append(result)

    # Sort by score descending
    scored.sort(key=lambda x: x["score"], reverse=True)

    # Stage 3: MMR diversity re-ranking
    final = mmr_rerank(scored, limit)

    # Sort final result (MMR may slightly reorder)
    final.sort(key=lambda x: x["score"], reverse=True)

    return final[:limit]


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4: Adaptive weight update (per-user online learning)
# ─────────────────────────────────────────────────────────────────────────────

def update_user_weights(conn, user_id: str, event_type: str,
                        chosen_scheme_breakdown: dict,
                        shown_schemes_breakdowns: list):
    """
    Multiplicative-weights update after an interaction event.

    For chosen scheme c with reward r:
      Δ_k = f_k(c) − mean_k(S)   [feature deviation from shown mean]
      w_k ← w_k * exp(η * r * Δ_k)
      clip to [MIN_WEIGHT, MAX_WEIGHT]
      renormalize so Σ w = 1

    Also updates category_affinity with EMA.
    """
    reward = EVENT_REWARDS.get(event_type, 0)
    if reward == 0:
        return  # No learning signal from pure impressions

    # Get current weights
    weights = get_user_weights(conn, user_id)
    t = weights.get("_interaction_count", 0)

    # Decaying learning rate
    eta = LEARNING_RATE / (1 + t / LR_DECAY_TAU)

    # Compute mean feature values across shown schemes
    feature_keys = [k for k in chosen_scheme_breakdown if k.startswith("f")]
    if not feature_keys or not shown_schemes_breakdowns:
        return

    mean_features = {}
    for fk in feature_keys:
        vals = [b.get(fk, 0.5) for b in shown_schemes_breakdowns]
        mean_features[fk] = sum(vals) / len(vals)

    # Map feature names to weight keys
    FEAT_TO_WEIGHT = {
        "f1_goal_relevance":    "f1_goal_relevance",
        "f2_financial_benefit": "f2_financial_benefit",
        "f3_urgency_match":     "f3_urgency_match",
        "f4_eligibility_margin": "f4_eligibility_margin",
        "f5_ease_of_access":    "f5_ease_of_access",
        "f6_coverage_priority": "f6_coverage_priority",
        "f7_category_affinity": "f7_category_affinity",
        "f8_recency_deadline":  "f8_recency_deadline",
    }

    new_weights = {}
    weight_keys_map = {
        "f1_goal_relevance":    "w_goal_relevance",
        "f2_financial_benefit": "w_financial_benefit",
        "f3_urgency_match":     "w_urgency_match",
        "f4_eligibility_margin": "w_eligibility_margin",
        "f5_ease_of_access":    "w_ease_of_access",
        "f6_coverage_priority": "w_coverage_priority",
        "f7_category_affinity": "w_category_affinity",
        "f8_recency_deadline":  "w_recency_deadline",
    }

    for fk in feature_keys:
        wkey = fk  # e.g. "f1_goal_relevance"
        if wkey not in weights:
            continue
        delta_k = chosen_scheme_breakdown.get(fk, 0.5) - mean_features.get(fk, 0.5)
        new_w = weights[wkey] * math.exp(eta * reward * delta_k)
        new_w = max(MIN_WEIGHT, min(MAX_WEIGHT, new_w))
        new_weights[wkey] = new_w

    if not new_weights:
        return

    # Normalize so sum = 1
    total = sum(new_weights.values())
    new_weights = {k: v / total for k, v in new_weights.items()}

    # Regularize toward global prior (prevent drift)
    for k in new_weights:
        new_weights[k] = (1 - GLOBAL_ALPHA) * new_weights[k] + \
                         GLOBAL_ALPHA * GLOBAL_WEIGHTS.get(k, 0.125)
    total2 = sum(new_weights.values())
    new_weights = {k: round(v / total2, 6) for k, v in new_weights.items()}

    # Write back to DB
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO user_weight_profiles (
                user_id,
                w_goal_relevance, w_financial_benefit, w_urgency_match, w_eligibility_margin,
                w_ease_of_access, w_coverage_priority, w_category_affinity, w_recency_deadline,
                interaction_count, last_retrained_at, version
            ) VALUES (
                %(user_id)s,
                %(f1)s, %(f2)s, %(f3)s, %(f4)s, %(f5)s, %(f6)s, %(f7)s, %(f8)s,
                1, NOW(), 1
            )
            ON CONFLICT (user_id) DO UPDATE SET
                w_goal_relevance    = %(f1)s,
                w_financial_benefit = %(f2)s,
                w_urgency_match     = %(f3)s,
                w_eligibility_margin = %(f4)s,
                w_ease_of_access    = %(f5)s,
                w_coverage_priority = %(f6)s,
                w_category_affinity = %(f7)s,
                w_recency_deadline  = %(f8)s,
                interaction_count   = user_weight_profiles.interaction_count + 1,
                last_retrained_at   = NOW(),
                version             = user_weight_profiles.version + 1
        """, {
            "user_id": user_id,
            "f1": new_weights.get("f1_goal_relevance", weights.get("f1_goal_relevance", 0.25)),
            "f2": new_weights.get("f2_financial_benefit", weights.get("f2_financial_benefit", 0.20)),
            "f3": new_weights.get("f3_urgency_match", weights.get("f3_urgency_match", 0.15)),
            "f4": new_weights.get("f4_eligibility_margin", weights.get("f4_eligibility_margin", 0.12)),
            "f5": new_weights.get("f5_ease_of_access", weights.get("f5_ease_of_access", 0.08)),
            "f6": new_weights.get("f6_coverage_priority", weights.get("f6_coverage_priority", 0.08)),
            "f7": new_weights.get("f7_category_affinity", weights.get("f7_category_affinity", 0.07)),
            "f8": new_weights.get("f8_recency_deadline", weights.get("f8_recency_deadline", 0.05)),
        })
    conn.commit()
    log.info(f"[Weights] Updated weights for user {user_id[:8]} after {event_type} (t={t})")


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI service
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="ArthSathi Recommendation Engine", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class RecommendRequest(BaseModel):
    user_id: str
    limit: int = Field(default=10, ge=1, le=20)
    item_type: str = Field(default="scheme")  # "scheme" | "insurance" | "both"
    session_id: Optional[str] = None

class EventRequest(BaseModel):
    user_id: str
    entity_type: str    # "scheme" | "insurance"
    entity_id: str
    event_type: str     # apply/save/click/listen/dismiss/impression
    rank_position: Optional[int] = None
    session_id: Optional[str] = None
    client_event_id: Optional[str] = None
    dwell_time_ms: Optional[int] = None
    score_breakdown: Optional[dict] = None


@app.get("/health")
def health():
    try:
        conn = get_db()
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) as cnt FROM schemes WHERE is_active = TRUE")
            scheme_count = cur.fetchone()["cnt"]
            cur.execute("SELECT COUNT(*) as cnt FROM insurance_policies WHERE is_active = TRUE")
            ins_count = cur.fetchone()["cnt"]
        conn.close()
        return {
            "status": "ok",
            "schemes": scheme_count,
            "insurance_policies": ins_count,
            "engine_version": "2.0",
            "features": 8,
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/recommend")
def recommend_endpoint(req: RecommendRequest):
    """
    Get personalised recommendations for a user.
    Returns up to `limit` schemes and/or insurance policies.
    """
    conn = get_db()
    try:
        # Fetch user + financial profile
        with conn.cursor() as cur:
            cur.execute("""
                SELECT fp.*, u.state, u.district, u.preferred_language,
                       u.interaction_mode, u.residence_type
                FROM financial_profiles fp
                JOIN users u ON u.id = fp.user_id
                WHERE fp.user_id = %s
            """, (req.user_id,))
            row = cur.fetchone()

        if not row:
            raise HTTPException(status_code=404, detail="Profile not found. Complete onboarding first.")

        profile = dict(row)

        # Get user weights
        weights = get_user_weights(conn, req.user_id)

        results = {}
        session_id = req.session_id or hashlib.md5(
            f"{req.user_id}{time.time()}".encode()).hexdigest()[:16]

        if req.item_type in ("scheme", "both"):
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, source_id, name, description, benefits_text, eligibility_text,
                           level, applicable_states, ministry, category, sub_category, tags,
                           benefit_type, benefit_amount_min, benefit_amount_max, benefit_frequency,
                           application_url, documents_required, application_deadline,
                           min_age, max_age, eligible_genders, eligible_castes, eligible_occupations,
                           applicable_states AS eligible_states_exact,
                           eligible_ration_cards, eligible_residence,
                           min_annual_income, max_annual_income,
                           requires_bank_account, requires_aadhaar, requires_bpl,
                           requires_disability, requires_widow, requires_student, requires_farmer,
                           last_scraped_at
                    FROM schemes WHERE is_active = TRUE
                """)
                schemes = [dict(r) for r in cur.fetchall()]

            scheme_recs = run_recommendations(profile, schemes, weights, "scheme", req.limit)

            # Persist matches for interaction tracking
            for rank, rec in enumerate(scheme_recs[:req.limit]):
                try:
                    with conn.cursor() as cur:
                        cur.execute("""
                            INSERT INTO user_scheme_matches
                            (user_id, scheme_id, score, rank_position, score_breakdown,
                             eligibility_status, missing_criteria, explain_text,
                             session_id, profile_snapshot, weights_snapshot)
                            VALUES (%s, %s, %s, %s, %s::jsonb, %s::eligibility_status_t, %s,
                                    %s, %s, %s::jsonb, %s::jsonb)
                            ON CONFLICT (user_id, scheme_id, session_id) DO NOTHING
                        """, (
                            req.user_id, rec["id"], rec["score"], rank + 1,
                            json.dumps(rec.get("score_breakdown")),
                            rec.get("eligibility_status", "likely"),
                            rec.get("missing_criteria"),
                            rec.get("explain_text"),
                            session_id,
                            json.dumps({k: str(v) for k, v in profile.items() if v is not None}),
                            json.dumps({k: v for k, v in weights.items() if not k.startswith("_")}),
                        ))
                    conn.commit()
                except Exception:
                    conn.rollback()

            results["schemes"] = scheme_recs

        if req.item_type in ("insurance", "both"):
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, source_id, name, description, eligibility_text,
                           category, tags, benefit_type, annual_premium, coverage_amount,
                           coverage_type, application_url, documents_required,
                           min_age, max_age, eligible_genders, eligible_castes,
                           eligible_ration_cards, min_annual_income, max_annual_income,
                           requires_bank_account, requires_aadhaar, requires_jan_dhan,
                           requires_bpl, last_scraped_at,
                           annual_premium AS benefit_amount_max,
                           coverage_amount AS benefit_amount_min
                    FROM insurance_policies WHERE is_active = TRUE
                """)
                policies = [dict(r) for r in cur.fetchall()]

            ins_recs = run_recommendations(profile, policies, weights, "insurance", 5)
            results["insurance"] = ins_recs

        results["session_id"] = session_id
        results["weights_used"] = {k: v for k, v in weights.items() if not k.startswith("_")}
        results["profile_summary"] = {
            "age": profile.get("age"),
            "state": profile.get("state"),
            "caste": profile.get("caste_category"),
            "occupation": profile.get("occupation"),
            "annual_income": profile.get("annual_income"),
            "goal": profile.get("financial_goal"),
        }

        return results

    finally:
        conn.close()


@app.post("/event")
def log_event(req: EventRequest, background: BackgroundTasks):
    """
    Log a user interaction event and trigger per-user weight update.
    Idempotent via client_event_id.
    """
    conn = get_db()
    try:
        client_id = req.client_event_id
        if not client_id:
            client_id = hashlib.md5(
                f"{req.user_id}{req.entity_id}{req.event_type}{time.time()}".encode()
            ).hexdigest()

        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO interaction_events
                (user_id, session_id, event_type, entity_type, entity_id,
                 rank_position, score_breakdown, dwell_time_ms, client_event_id, device_type)
                VALUES (%s, %s, %s::match_action_t, %s, %s, %s, %s::jsonb, %s, %s, 'web')
                ON CONFLICT (client_event_id) DO NOTHING
            """, (
                req.user_id, req.session_id,
                req.event_type, req.entity_type, req.entity_id,
                req.rank_position,
                json.dumps(req.score_breakdown) if req.score_breakdown else None,
                req.dwell_time_ms, client_id,
            ))
        conn.commit()

        # Trigger adaptive learning in background
        if req.score_breakdown and req.event_type in EVENT_REWARDS:
            background.add_task(
                _background_weight_update,
                req.user_id, req.event_type, req.score_breakdown, req.session_id
            )

        return {"success": True, "client_event_id": client_id}

    except Exception as e:
        conn.rollback()
        log.error(f"Event log error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()


def _background_weight_update(user_id: str, event_type: str,
                               chosen_breakdown: dict, session_id: str):
    """Background task: update per-user weights after interaction."""
    conn = get_db()
    try:
        # Fetch breakdowns of all schemes shown in this session
        with conn.cursor() as cur:
            cur.execute("""
                SELECT score_breakdown FROM user_scheme_matches
                WHERE user_id = %s AND session_id = %s AND score_breakdown IS NOT NULL
            """, (user_id, session_id))
            rows = cur.fetchall()
        shown_breakdowns = [r["score_breakdown"] for r in rows if r["score_breakdown"]]

        update_user_weights(conn, user_id, event_type, chosen_breakdown, shown_breakdowns)
    except Exception as e:
        log.error(f"Background weight update error: {e}")
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# CLI test mode
# ─────────────────────────────────────────────────────────────────────────────

def run_cli_test():
    """Test the engine against synthetic profiles and print results."""
    SYNTHETIC_PROFILES = [
        {
            "name": "Ramesh — SC farmer, 35yo, Maharashtra, Rs.60K/yr",
            "profile": {
                "age": 35, "gender": "male", "caste_category": "sc",
                "occupation": "farmer", "state": "Maharashtra",
                "annual_income": 60000, "monthly_income": 5000,
                "total_debts": 20000, "num_dependents": 3,
                "has_bank_account": True, "has_jan_dhan_account": True,
                "has_aadhaar": True, "is_bpl": True,
                "ration_card_type": "phh",
                "land_holding_acres": 2.0,
                "financial_goal": "farming",
                "goal_description": "buy seeds and fertilizer for kharif",
                "goal_amount": 15000, "urgency_level": 4,
            },
        },
        {
            "name": "Savita — Daily wage labourer, OBC woman, 28yo, UP, Rs.36K/yr",
            "profile": {
                "age": 28, "gender": "female", "caste_category": "obc",
                "occupation": "daily_wage", "state": "Uttar Pradesh",
                "annual_income": 36000, "monthly_income": 3000,
                "total_debts": 5000, "num_dependents": 2,
                "has_bank_account": True, "has_jan_dhan_account": True,
                "has_aadhaar": False, "is_bpl": True,
                "ration_card_type": "aay", "is_widow": True,
                "financial_goal": "housing",
                "goal_description": "need pucca house",
                "goal_amount": 130000, "urgency_level": 5,
            },
        },
        {
            "name": "Arjun — SC student, 20yo, Tamil Nadu, no income",
            "profile": {
                "age": 20, "gender": "male", "caste_category": "sc",
                "occupation": "student", "state": "Tamil Nadu",
                "annual_income": 0, "monthly_income": 0,
                "total_debts": 0, "num_dependents": 0,
                "has_bank_account": True, "has_aadhaar": True,
                "is_student": True, "education_level": "higher_secondary",
                "ration_card_type": "phh",
                "financial_goal": "education",
                "goal_description": "scholarship for college",
                "goal_amount": 50000, "urgency_level": 3,
            },
        },
        {
            "name": "Rehman — ST fisherman, 45yo, Kerala, disabled",
            "profile": {
                "age": 45, "gender": "male", "caste_category": "st",
                "occupation": "fisherman", "state": "Kerala",
                "annual_income": 72000, "monthly_income": 6000,
                "total_debts": 30000, "num_dependents": 4,
                "has_bank_account": True, "has_aadhaar": True,
                "has_disability": True, "disability_type": "locomotor",
                "disability_percentage": 45,
                "is_bpl": False, "ration_card_type": "nphh",
                "financial_goal": "health",
                "goal_description": "medical treatment and prosthetics",
                "goal_amount": 80000, "urgency_level": 4,
            },
        },
        {
            "name": "Meena — widowed OBC tailor, 38yo, Tamil Nadu, Rs.48K/yr",
            "profile": {
                "age": 38, "gender": "female", "caste_category": "obc",
                "occupation": "artisan", "state": "Tamil Nadu",
                "annual_income": 48000, "monthly_income": 4000,
                "total_debts": 15000, "num_dependents": 3,
                "has_bank_account": True, "has_aadhaar": True,
                "is_widow": True, "num_girl_children": 2, "has_girl_child": True,
                "ration_card_type": "phh",
                "financial_goal": "employment",
                "goal_description": "grow tailoring business",
                "goal_amount": 50000, "urgency_level": 3,
            },
        },
    ]

    conn = get_db()
    weights_global = dict(GLOBAL_WEIGHTS)
    weights_global["_interaction_count"] = 0
    weights_global["_category_affinity"] = {}

    with conn.cursor() as cur:
        cur.execute("SELECT id, source_id, name, category, tags, benefit_type, benefit_amount_min, benefit_amount_max, min_age, max_age, eligible_genders, eligible_castes, eligible_occupations, eligible_ration_cards, eligible_residence, applicable_states, max_annual_income, requires_bank_account, requires_aadhaar, requires_bpl, requires_disability, requires_widow, requires_student, requires_farmer, last_scraped_at FROM schemes WHERE is_active = TRUE")
        schemes = [dict(r) for r in cur.fetchall()]

        cur.execute("SELECT id, source_id, name, category, tags, benefit_type, annual_premium, coverage_amount, min_age, max_age, eligible_genders, eligible_castes, eligible_ration_cards, max_annual_income, requires_bank_account, requires_jan_dhan, requires_bpl, last_scraped_at, coverage_amount AS benefit_amount_max FROM insurance_policies WHERE is_active = TRUE")
        insurance = [dict(r) for r in cur.fetchall()]
    conn.close()

    print(f"\n{'='*72}")
    print("ArthSathi Recommendation Engine v2.0 — Synthetic Profile Tests")
    print(f"Active weights: {weights_global}")
    print(f"Total schemes: {len(schemes)}, Insurance: {len(insurance)}")
    print(f"{'='*72}\n")

    for test in SYNTHETIC_PROFILES:
        profile = test["profile"]
        print(f"\n{'─'*72}")
        print(f"Profile: {test['name']}")
        print(f"Goal: \"{profile.get('goal_description')}\" (urgency={profile.get('urgency_level')}/5)")

        recs = run_recommendations(profile, schemes, weights_global, "scheme", 10)
        ins_recs = run_recommendations(profile, insurance, weights_global, "insurance", 3)

        print(f"\n  Top Schemes ({len(recs)} eligible of {len(schemes)}):")
        for i, r in enumerate(recs[:5], 1):
            name = (r.get("name") or {}).get("en", "")[:55]
            eligibility = r.get("eligibility_status", "?")
            explore = " [EXPLORE]" if r.get("_is_exploration") else ""
            print(f"    {i}. [{r['score']:.4f}] ({eligibility}){explore} {name}")
            bd = r.get("score_breakdown", {})
            top_feats = sorted(bd.items(), key=lambda x: x[1], reverse=True)[:3]
            print(f"       {' | '.join(f'{k}={v:.2f}' for k,v in top_feats)}")
            print(f"       {r.get('explain_text', '')[:80]}")

        print(f"\n  Top Insurance ({len(ins_recs)} eligible of {len(insurance)}):")
        for i, r in enumerate(ins_recs, 1):
            name = (r.get("name") or {}).get("en", "")[:55]
            prem = r.get("annual_premium")
            prem_str = f"₹{prem}" if prem else "variable"
            print(f"    {i}. [{r['score']:.4f}] {name} | premium: {prem_str}")

    print(f"\n{'='*72}\n")


if __name__ == "__main__":
    import sys
    if "--test" in sys.argv:
        run_cli_test()
    else:
        import uvicorn
        uvicorn.run("recommendation_engine:app", host="127.0.0.1", port=8001, reload=False)

