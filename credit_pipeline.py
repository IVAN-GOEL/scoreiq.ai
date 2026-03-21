# =============================================================================
# credit_pipeline.py
# ScoreIQ — Full ML pipeline
#
# Flow:
#   Setu AA transactions  ──┐
#   Self-declared assets  ──┤── feature engineering ── RandomForest ── SHAP
#   PDF bank statement    ──┘
#
# Run training:  python credit_pipeline.py --train
# Run smoke test: python credit_pipeline.py
# =============================================================================

import io
import os
import pickle
import logging
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

MODEL_PATH = "data/model.pkl"


# =============================================================================
# SECTION 1 — DATA CLASSES
# =============================================================================

@dataclass
class Asset:
    """
    One asset declared by the applicant on the form.

    type options:
        real_estate | gold | fd | mutual_fund | epf_ppf | vehicle
    """
    type: str
    declared_value: float
    outstanding_loan: float = 0.0
    description: str = ""


@dataclass
class ApplicantInput:
    """
    Everything collected from one applicant.
      transactions         — raw list from Setu AA /fi/fetch OR extracted by pdf_parser
      assets               — self-declared assets from the frontend form
      loan_amount_requested — optional, used for DTI ratio checks in future
    """
    transactions: list
    assets: list = field(default_factory=list)
    loan_amount_requested: float = 0.0


# =============================================================================
# SECTION 2 — TRANSACTION FEATURE EXTRACTION
# Converts 100–300 raw bank rows into 8 meaningful numbers.
# =============================================================================

EMI_PATTERN    = r"EMI|LOAN|REPAY|NACH|ECS"
SALARY_PATTERN = r"SALARY|SAL|PAYROLL|NEFT CR|NEFT/SAL"


def extract_transaction_features(transactions: list) -> dict:
    """
    Input : raw transactions list (from Setu AA or pdf_parser)
    Output: 8 numerical features ready for the ML model

    Feature descriptions
    --------------------
    avg_monthly_income    — average credits per calendar month
    avg_monthly_expenses  — average debits per calendar month
    savings_ratio         — (income − expenses) / income  [clamped −1..1]
    income_volatility     — coefficient of variation of monthly credits
                            (near 0 = stable salary, >0.5 = irregular/gig)
    emi_count             — number of EMI / NACH / loan repayment transactions
    low_balance_dips      — times the account balance dropped below ₹500
    salary_regularity     — std/mean of monthly salary credits
                            (near 0 = same amount every month)
    income_sufficient     — 1 if avg income > avg expenses, else 0
    """
    if not transactions:
        logger.warning("No transactions — returning zero features")
        return {k: 0.0 for k in [
            "avg_monthly_income", "avg_monthly_expenses", "savings_ratio",
            "income_volatility", "emi_count", "low_balance_dips",
            "salary_regularity", "income_sufficient",
        ]}

    try:
        df = pd.DataFrame(transactions)
        df["amount"]  = pd.to_numeric(df["amount"],  errors="coerce")
        df["date"]    = pd.to_datetime(df["date"],   errors="coerce")
        df["balance"] = pd.to_numeric(df.get("balance", 0), errors="coerce").fillna(0)

        # Drop rows with unparseable dates or amounts
        bad = df["date"].isna() | df["amount"].isna()
        if bad.any():
            logger.warning(f"Dropping {bad.sum()} malformed rows")
            df = df[~bad]

        if df.empty:
            raise ValueError("All rows were invalid")

    except Exception as exc:
        logger.error(f"Transaction parsing failed: {exc}")
        raise ValueError(f"Invalid transaction data: {exc}") from exc

    df = df.set_index("date").sort_index()
    df["type"]      = df["type"].str.upper()
    df["narration"] = df.get("narration", pd.Series("", index=df.index)).fillna("").str.upper()

    credits = df[df["type"] == "CREDIT"]["amount"]
    debits  = df[df["type"] == "DEBIT"]["amount"]

    # Monthly aggregates
    monthly_in  = credits.resample("ME").sum()
    monthly_out = debits.resample("ME").sum()
    avg_in      = float(monthly_in.mean())  if len(monthly_in)  > 0 else 0.0
    avg_out     = float(monthly_out.mean()) if len(monthly_out) > 0 else 0.0

    # Savings ratio — clamped to [−1, 1]
    total_in  = float(credits.sum())
    total_out = float(debits.sum())
    savings_ratio = (1 - total_out / total_in) if total_in > 0 else 0.0
    savings_ratio = round(max(-1.0, min(1.0, savings_ratio)), 4)

    # Income volatility — coefficient of variation
    if len(credits) < 4:
        income_vol = 0.8   # too few rows to measure — treat as risky
    elif credits.mean() > 0:
        income_vol = float(credits.std() / credits.mean())
    else:
        income_vol = 1.0

    # EMI count
    emi_count = int(df["narration"].str.contains(EMI_PATTERN, case=False, na=False).sum())

    # Low-balance dips
    low_dips = int((df["balance"] < 500).sum())

    # Salary regularity
    credit_rows  = df[df["type"] == "CREDIT"]
    salary_rows  = credit_rows[credit_rows["narration"].str.contains(SALARY_PATTERN, case=False, na=False)]
    monthly_sal  = salary_rows["amount"].resample("ME").sum()
    if len(monthly_sal) < 2 or monthly_sal.mean() == 0:
        salary_reg = 1.0   # undetectable salary = maximum irregularity
    else:
        salary_reg = float(monthly_sal.std() / monthly_sal.mean())

    return {
        "avg_monthly_income":   round(avg_in, 2),
        "avg_monthly_expenses": round(avg_out, 2),
        "savings_ratio":        savings_ratio,
        "income_volatility":    round(income_vol, 4),
        "emi_count":            emi_count,
        "low_balance_dips":     low_dips,
        "salary_regularity":    round(salary_reg, 4),
        "income_sufficient":    1 if avg_in > avg_out else 0,
    }


# =============================================================================
# SECTION 3 — ASSET SCORING
# Weights each asset class and penalises heavily-mortgaged assets.
# =============================================================================

# Appreciation/reliability weights per asset class
ASSET_WEIGHTS = {
    "real_estate": 0.90,   # appreciates ~8–12% p.a., high liquidity
    "gold":        0.85,   # reliable store of value
    "fd":          0.95,   # near-liquid, guaranteed return
    "mutual_fund": 0.70,   # market-linked, volatile
    "epf_ppf":     0.60,   # locked — penalised for illiquidity
    "vehicle":     0.30,   # depreciates ~15–20% p.a.
}

EQUITY_DANGER_THRESHOLD = 0.40   # below 40% equity → danger zone


def score_single_asset(asset: Asset) -> dict:
    """
    Computes the net weighted value of one asset.

    Logic:
      1. Look up the asset-class weight (appreciation potential).
      2. equity = declared_value − outstanding_loan
      3. equity_ratio = equity / declared_value
      4. If equity_ratio < 40%, apply a proportional mortgage penalty.
      5. net_value = equity × weight × penalty
    """
    # Sanitise inputs
    declared = max(0.0, float(asset.declared_value))
    loan     = max(0.0, float(asset.outstanding_loan))

    w            = ASSET_WEIGHTS.get(str(asset.type), 0.50)
    equity       = max(0.0, declared - loan)
    equity_ratio = equity / declared if declared > 0 else 0.0

    # Smooth mortgage penalty: full credit above threshold, scales to 0 below
    penalty = 1.0 if equity_ratio >= EQUITY_DANGER_THRESHOLD else (
        equity_ratio / EQUITY_DANGER_THRESHOLD
    )

    return {
        "type":             asset.type,
        "declared_value":   declared,
        "outstanding_loan": loan,
        "equity":           round(equity, 2),
        "equity_ratio":     round(equity_ratio, 4),
        "net_contribution": round(equity * w * penalty, 2),
    }


def extract_asset_features(assets: list) -> dict:
    """
    Aggregates all assets into 4 features for the model.

    Features:
      total_asset_value  — sum of net weighted value across all assets
      asset_diversity    — number of distinct asset types
      vehicle_ratio      — fraction of total value in depreciating assets
      mortgage_burden    — total outstanding loans
    """
    if not assets:
        return {"total_asset_value": 0.0, "asset_diversity": 0,
                "vehicle_ratio": 0.0, "mortgage_burden": 0.0}

    scored    = [score_single_asset(a) for a in assets]
    total_net = sum(s["net_contribution"] for s in scored)
    total_raw = sum(float(a.declared_value) for a in assets)
    veh_raw   = sum(float(a.declared_value) for a in assets if a.type == "vehicle")
    total_due = sum(float(a.outstanding_loan) for a in assets)

    return {
        "total_asset_value": round(total_net, 2),
        "asset_diversity":   len({a.type for a in assets}),
        "vehicle_ratio":     round(veh_raw / total_raw, 4) if total_raw > 0 else 0.0,
        "mortgage_burden":   round(total_due, 2),
    }


# =============================================================================
# SECTION 4 — FEATURE VECTOR
# 12 features total: 8 from transactions + 4 from assets
# =============================================================================

FEATURE_NAMES = [
    # From bank transactions
    "avg_monthly_income",
    "avg_monthly_expenses",
    "savings_ratio",
    "income_volatility",
    "emi_count",
    "low_balance_dips",
    "salary_regularity",
    "income_sufficient",
    # From declared assets
    "total_asset_value",
    "asset_diversity",
    "vehicle_ratio",
    "mortgage_burden",
]

# Human-readable labels shown in the UI explanation panel
FEATURE_LABELS = {
    "avg_monthly_income":   "Average monthly income",
    "avg_monthly_expenses": "Average monthly expenses",
    "savings_ratio":        "Savings ratio",
    "income_volatility":    "Income stability",
    "emi_count":            "Active EMI / loan payments",
    "low_balance_dips":     "Near-zero balance incidents",
    "salary_regularity":    "Salary regularity",
    "income_sufficient":    "Income exceeds expenses",
    "total_asset_value":    "Net weighted asset value",
    "asset_diversity":      "Asset diversity",
    "vehicle_ratio":        "Depreciating asset ratio",
    "mortgage_burden":      "Total outstanding asset loans",
}


def build_feature_vector(applicant: ApplicantInput) -> dict:
    """
    Master feature builder — combines transaction + asset features
    into one flat dict aligned to FEATURE_NAMES.
    """
    tx_feat    = extract_transaction_features(applicant.transactions)
    asset_feat = extract_asset_features(applicant.assets)
    combined   = {**tx_feat, **asset_feat}
    return {k: combined.get(k, 0.0) for k in FEATURE_NAMES}


# =============================================================================
# SECTION 5 — SCORE GENERATION
# Converts default probability to 300–850 score + risk band
# =============================================================================

def probability_to_score(default_prob: float) -> dict:
    """
    Maps default probability [0, 1] → credit score [300, 850].
    Formula: score = 850 − (prob × 550)

    Bands:
      750–850  Excellent   (green)
      670–749  Good        (teal / blue)
      580–669  Fair        (orange / amber)
      300–579  Poor        (red)
    """
    score = max(300, min(850, int(round(850 - default_prob * 550))))

    if score >= 750:
        band, color = "Excellent", "green"
    elif score >= 670:
        band, color = "Good", "blue"
    elif score >= 580:
        band, color = "Fair", "amber"
    else:
        band, color = "Poor", "red"

    return {
        "score":               score,
        "band":                band,
        "color":               color,
        "default_probability": round(default_prob, 4),
    }


# =============================================================================
# SECTION 6 — SHAP EXPLANATION
# Returns top N factors driving the score in plain language
# =============================================================================

def explain_score(model, feature_vector: dict, top_n: int = 5) -> list:
    """
    Uses SHAP TreeExplainer to rank which features drove the prediction.

    Returns list of dicts like:
        {
          feature:    "savings_ratio",
          label:      "Savings ratio",
          shap_value: 0.12,
          direction:  "positive",   # helped applicant
          impact_pts: 66            # ≈ score-point equivalent
        }

    impact_pts = abs(shap_value) × 550  (mirrors our score scale)
    """
    try:
        explainer = shap.TreeExplainer(model)
        row       = pd.DataFrame([feature_vector])[FEATURE_NAMES]
        raw_shap  = explainer.shap_values(row)
    except Exception as exc:
        logger.error(f"SHAP failed: {exc}")
        return []

    # Binary classifier: shap_values returns [class_0, class_1]
    # We want class_1 (default probability)
    if isinstance(raw_shap, list) and len(raw_shap) > 1:
        vals = raw_shap[1][0]
    elif isinstance(raw_shap, np.ndarray):
        vals = raw_shap[0] if raw_shap.ndim > 1 else raw_shap
    else:
        vals = []

    results = []
    for feat, sv in zip(FEATURE_NAMES, vals):
        sv = float(np.array(sv).flatten()[0]) if hasattr(sv, "__len__") else float(sv)
        # A positive SHAP value → higher default probability → NEGATIVE for applicant
        # A negative SHAP value → lower default probability → POSITIVE for applicant
        direction  = "negative" if sv > 0 else "positive"
        impact_pts = int(round(abs(sv) * 550))
        results.append({
            "feature":    feat,
            "label":      FEATURE_LABELS.get(feat, feat),
            "shap_value": round(sv, 5),
            "direction":  direction,
            "impact_pts": impact_pts,
        })

    results.sort(key=lambda x: x["impact_pts"], reverse=True)
    return results[:top_n]


# =============================================================================
# SECTION 7 — FULL INFERENCE
# Single public function: ApplicantInput → score + SHAP explanation
# =============================================================================


# =============================================================================
# SECTION 7B — CLAUDE AI ENHANCEMENT
# Uses Claude to generate intelligent narrative on top of RF score + SHAP
# =============================================================================

import os, json, httpx as _httpx

CLAUDE_SCORING_PROMPT = """You are a senior credit analyst at a fintech company in India.

You have been given the output of a RandomForest credit scoring model for an applicant.
Your job is to:
1. Validate the score makes sense given the features
2. Write a clear 2-3 sentence explanation in plain English
3. Identify the top 3 things the applicant can do to improve their score
4. Flag any red flags or positive signals the model may have missed

Return ONLY a JSON object, no markdown, no explanation:
{
  "validated_score": <int 300-850, adjust by max ±30 if you disagree with RF>,
  "explanation": "<2-3 sentence plain English summary of why this person got this score>",
  "improvement_tips": [
    "<specific actionable tip 1>",
    "<specific actionable tip 2>",
    "<specific actionable tip 3>"
  ],
  "red_flags": ["<flag1 if any>"],
  "positive_signals": ["<signal1>", "<signal2>"],
  "analyst_note": "<one sentence analyst observation>"
}"""


def claude_enhance_score(features: dict, rf_result: dict, top_factors: list) -> dict:
    """
    Send RF score + features to Claude for intelligent enhancement.
    Returns enhanced explanation, tips, flags.
    Falls back gracefully if Claude API unavailable.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return {}

    # Build context for Claude
    context = f"""
RF Credit Score: {rf_result['score']} / 850
Risk Band: {rf_result['band']}
Default Probability: {rf_result['default_probability']:.1%}

Key Financial Features:
- Monthly Income: ₹{features.get('avg_monthly_income', 0):,.0f}
- Monthly Expenses: ₹{features.get('avg_monthly_expenses', 0):,.0f}
- Savings Ratio: {features.get('savings_ratio', 0):.1%}
- Income Volatility: {features.get('income_volatility', 0):.2f} (0=stable, 1=volatile)
- EMI Payments: {features.get('emi_count', 0)} transactions detected
- Low Balance Incidents: {features.get('low_balance_dips', 0)} times
- Salary Regularity: {features.get('salary_regularity', 0):.2f} (0=perfect, 1=irregular)
- Income > Expenses: {'Yes' if features.get('income_sufficient', 0) else 'No'}
- Net Asset Value: ₹{features.get('total_asset_value', 0):,.0f}
- Asset Diversity: {features.get('asset_diversity', 0)} types
- Vehicle Ratio: {features.get('vehicle_ratio', 0):.1%} of assets in vehicles
- Mortgage Burden: ₹{features.get('mortgage_burden', 0):,.0f}

Top SHAP Drivers:
""" + "\n".join([f"- {f['label']}: {f['direction']} ({f['impact_pts']} pts)" for f in top_factors[:5]])

    try:
        resp = _httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 1024,
                "messages": [{
                    "role": "user",
                    "content": CLAUDE_SCORING_PROMPT + "\n\nApplicant Data:\n" + context
                }]
            },
            timeout=30.0,
        )
        resp.raise_for_status()
        text = resp.json()["content"][0]["text"]
        # Strip markdown fences if present
        import re as _re
        text = _re.sub(r"```(?:json)?\s*", "", text).strip()
        return json.loads(text)
    except Exception as exc:
        logger.warning(f"Claude enhancement failed: {exc}")
        return {}

def assess_applicant(applicant: ApplicantInput, model) -> dict:
    """
    End-to-end assessment.

    Returns:
        score               int        300–850
        band                str        Excellent | Good | Fair | Poor
        color               str        green | blue | amber | red
        default_probability float      0–1
        risk_band           str        alias for band (for frontend compatibility)
        probability_of_default float   alias for default_probability
        features            dict       the 12-feature vector
        top_factors         list[dict] SHAP explanation (top 5 drivers)
        shap_values         list[dict] same as top_factors (frontend compatibility)
        explanation         str        plain-English summary
    """
    logger.info(f"Assessing: {len(applicant.transactions)} txns, {len(applicant.assets)} assets")

    features     = build_feature_vector(applicant)
    row          = pd.DataFrame([features])[FEATURE_NAMES]
    default_prob = float(model.predict_proba(row)[0][1])
    score_result = probability_to_score(default_prob)
    top_factors  = explain_score(model, features)

    # ── Claude AI enhancement ────────────────────────────────────────────────
    claude = claude_enhance_score(features, score_result, top_factors)

    # Use Claude's validated score if it adjusted (max ±30 pts)
    final_score = claude.get("validated_score", score_result["score"])
    final_score = max(300, min(850, final_score))  # clamp
    if abs(final_score - score_result["score"]) > 30:
        final_score = score_result["score"]  # ignore if Claude went too far

    # Use Claude's explanation if available, else build from SHAP
    top_driver  = top_factors[0]["label"] if top_factors else "n/a"
    explanation = claude.get("explanation") or (
        f"Estimated default probability is {default_prob:.1%}. "
        f"The strongest driver in this decision is '{top_driver}'."
    )

    return {
        **score_result,
        "score":                 final_score,
        "risk_band":             score_result["band"],
        "probability_of_default": default_prob,
        "features":              features,
        "top_factors":           top_factors,
        "shap_values":           [
            {"feature": f["feature"], "value": -f["shap_value"]}
            for f in top_factors
        ],
        "explanation":       explanation,
        "improvement_tips":  claude.get("improvement_tips", []),
        "red_flags":         claude.get("red_flags", []),
        "positive_signals":  claude.get("positive_signals", []),
        "analyst_note":      claude.get("analyst_note", ""),
        "ai_enhanced":       bool(claude),
    }


# =============================================================================
# SECTION 8 — MODEL TRAINING
# Generates synthetic 3-cluster data and trains RandomForest.
# Run: python credit_pipeline.py --train
# =============================================================================

def generate_synthetic_data(n: int = 3000) -> pd.DataFrame:
    """
    Generates labelled training data with 3 risk profiles:
      Cluster A — stable salaried workers     → low default risk
      Cluster B — gig / irregular income      → medium default risk
      Cluster C — financially stressed        → high default risk
    """
    rng   = np.random.default_rng(42)
    n_abc = n // 3

    # ── Cluster A: stable salaried (low risk) ──
    a_inc = rng.integers(40000, 150000, n_abc).astype(float)
    a_exp = a_inc * rng.uniform(0.30, 0.70, n_abc)
    A = pd.DataFrame({
        "avg_monthly_income":   a_inc,
        "avg_monthly_expenses": a_exp,
        "savings_ratio":        rng.uniform(0.25, 0.60, n_abc),
        "income_volatility":    rng.uniform(0.00, 0.15, n_abc),
        "emi_count":            rng.integers(0, 3, n_abc),
        "low_balance_dips":     rng.integers(0, 2, n_abc),
        "salary_regularity":    rng.uniform(0.00, 0.10, n_abc),
        "income_sufficient":    np.ones(n_abc, dtype=int),
        "total_asset_value":    rng.uniform(500000, 5000000, n_abc),
        "asset_diversity":      rng.integers(2, 6, n_abc),
        "vehicle_ratio":        rng.uniform(0.0, 0.3, n_abc),
        "mortgage_burden":      rng.uniform(0, 500000, n_abc),
        "default":              np.zeros(n_abc, dtype=int),
    })

    # ── Cluster B: gig / irregular (medium risk) ──
    b_inc = rng.integers(25000, 80000, n_abc).astype(float)
    b_exp = b_inc * rng.uniform(0.50, 0.85, n_abc)   # income still > expenses
    B = pd.DataFrame({
        "avg_monthly_income":   b_inc,
        "avg_monthly_expenses": b_exp,
        "savings_ratio":        rng.uniform(0.15, 0.50, n_abc),
        "income_volatility":    rng.uniform(0.25, 0.75, n_abc),
        "emi_count":            rng.integers(0, 3, n_abc),
        "low_balance_dips":     rng.integers(0, 4, n_abc),
        "salary_regularity":    rng.uniform(0.25, 0.65, n_abc),
        "income_sufficient":    np.ones(n_abc, dtype=int),
        "total_asset_value":    rng.uniform(50000, 800000, n_abc),
        "asset_diversity":      rng.integers(1, 4, n_abc),
        "vehicle_ratio":        rng.uniform(0.1, 0.5, n_abc),
        "mortgage_burden":      rng.uniform(0, 400000, n_abc),
        "default":              rng.integers(0, 2, n_abc),   # 50/50
    })

    # ── Cluster C: financially stressed (high risk) ──
    n_c   = n - 2 * n_abc
    c_inc = rng.integers(8000, 20000, n_c).astype(float)
    c_exp = c_inc * rng.uniform(1.1, 1.8, n_c)   # expenses > income
    C = pd.DataFrame({
        "avg_monthly_income":   c_inc,
        "avg_monthly_expenses": c_exp,
        "savings_ratio":        rng.uniform(-0.8, -0.1, n_c),
        "income_volatility":    rng.uniform(0.60, 1.50, n_c),
        "emi_count":            rng.integers(2, 8, n_c),
        "low_balance_dips":     rng.integers(4, 15, n_c),
        "salary_regularity":    rng.uniform(0.60, 1.20, n_c),
        "income_sufficient":    np.zeros(n_c, dtype=int),
        "total_asset_value":    rng.uniform(0, 50000, n_c),
        "asset_diversity":      rng.integers(0, 2, n_c),
        "vehicle_ratio":        rng.uniform(0.5, 1.0, n_c),
        "mortgage_burden":      rng.uniform(300000, 1500000, n_c),
        "default":              np.ones(n_c, dtype=int),
    })

    df = pd.concat([A, B, C], ignore_index=True)
    df["savings_ratio"]        = df["savings_ratio"].clip(-1, 1)
    df["avg_monthly_expenses"] = df["avg_monthly_expenses"].clip(lower=1000)
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


def train_and_save(model_path: str = MODEL_PATH) -> None:
    """
    Trains a RandomForestClassifier on synthetic data and saves to disk.
    Called once during Docker build via Dockerfile CMD, or manually with --train.
    """
    os.makedirs("data", exist_ok=True)
    logger.info("Generating synthetic training data (3 000 samples)…")
    df = generate_synthetic_data(3000)

    X = df[FEATURE_NAMES]
    y = df["default"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    logger.info("Training RandomForestClassifier (200 trees)…")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_tr, y_tr)

    logger.info("\n" + classification_report(y_te, model.predict(X_te)))

    with open(model_path, "wb") as fh:
        pickle.dump({"model": model, "features": FEATURE_NAMES}, fh)
    logger.info(f"Model saved → {model_path}")


def load_model(model_path: str = MODEL_PATH):
    """
    Loads the trained model from disk.
    If the file is missing, trains a fresh model automatically.
    This means Railway containers always have a working model on first boot.
    """
    if not os.path.exists(model_path):
        logger.warning(f"Model not found at {model_path} — training now…")
        train_and_save(model_path)

    with open(model_path, "rb") as fh:
        bundle = pickle.load(fh)

    logger.info(f"Model loaded from {model_path}")
    return bundle["model"]


# =============================================================================
# SECTION 9 — SAMPLE DATA (used by /assess/demo endpoint)
# =============================================================================

SAMPLE_TRANSACTIONS = [
    {"date": "2024-10-03", "amount": "45000", "type": "CREDIT",
     "narration": "NEFT/SALARY/ACME CORP",  "balance": "52000"},
    {"date": "2024-10-07", "amount": "12000", "type": "DEBIT",
     "narration": "EMI/HDFC BANK LOAN",     "balance": "40000"},
    {"date": "2024-10-15", "amount": "3200",  "type": "DEBIT",
     "narration": "UPI/SWIGGY FOOD",        "balance": "36800"},
    {"date": "2024-11-03", "amount": "45000", "type": "CREDIT",
     "narration": "NEFT/SALARY/ACME CORP",  "balance": "58000"},
    {"date": "2024-11-07", "amount": "12000", "type": "DEBIT",
     "narration": "EMI/HDFC BANK LOAN",     "balance": "46000"},
    {"date": "2024-11-20", "amount": "8500",  "type": "DEBIT",
     "narration": "RENT TRANSFER",          "balance": "37500"},
    {"date": "2024-12-03", "amount": "45000", "type": "CREDIT",
     "narration": "NEFT/SALARY/ACME CORP",  "balance": "61000"},
    {"date": "2024-12-10", "amount": "12000", "type": "DEBIT",
     "narration": "EMI/HDFC BANK LOAN",     "balance": "49000"},
    {"date": "2024-12-28", "amount": "480",   "type": "DEBIT",
     "narration": "AIRTEL RECHARGE",        "balance": "48520"},
]

SAMPLE_ASSETS = [
    Asset("real_estate",  3_500_000, 2_000_000, "2BHK flat"),
    Asset("gold",           250_000,         0, "Jewellery"),
    Asset("mutual_fund",    180_000,         0, "SIP portfolio"),
    Asset("vehicle",        600_000,   350_000, "Car loan"),
    Asset("fd",             100_000,         0, "Bank FD"),
]


# =============================================================================
# CLI ENTRYPOINT
# =============================================================================

if __name__ == "__main__":
    import sys

    if "--train" in sys.argv:
        train_and_save()
    else:
        print("Loading model (auto-trains if missing)…")
        mdl = load_model()

        applicant = ApplicantInput(
            transactions=SAMPLE_TRANSACTIONS,
            assets=SAMPLE_ASSETS,
            loan_amount_requested=500_000,
        )

        result = assess_applicant(applicant, mdl)

        print(f"\n{'='*52}")
        print(f"  Credit Score : {result['score']}  ({result['band']})")
        print(f"  Default Prob : {result['default_probability']:.1%}")
        print(f"{'='*52}")
        print("\nTop SHAP factors:")
        for f in result["top_factors"]:
            arrow = "+" if f["direction"] == "positive" else "-"
            print(f"  [{arrow}{f['impact_pts']:>3} pts]  {f['label']}")
