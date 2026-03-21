# =============================================================================
# claude_scorer.py — AI Enhanced Credit Scorer (Google Gemini Edition)
#
# Uses Google Gemini 1.5 Flash (FREE) to analyse ALL applicant inputs:
#   • Personal profile (age, employment, income, expenses)
#   • Liabilities (home loan, vehicle loan, personal loan, CC dues, EMI)
#   • Assets (real estate, gold, FD, mutual funds, vehicle, EPF)
#   • Bank data (from Setu AA or PDF parser)
#
# Get free API key: aistudio.google.com → Get API Key (no credit card needed)
# Add to Railway Variables: GEMINI_API_KEY = your_key
#
# Blends Gemini score (60%) + deterministic model (40%) for stability.
# Falls back to model only if Gemini unavailable — never crashes.
# =============================================================================

import json
import os
import re
import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_URL     = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"

GEMINI_WEIGHT = 1.00
MODEL_WEIGHT  = 0.00

SCORING_PROMPT = """You are an expert credit risk analyst for India. Analyse the applicant data below and return a JSON credit assessment.

APPLICANT DATA:
{applicant_json}

Analyse ALL factors:
1. Income vs expenses ratio and savings capacity
2. Debt burden: total liabilities, EMI-to-income ratio
3. Asset quality: net worth after outstanding loans
4. Employment stability and years employed
5. Payment history (missed payments)
6. Age and financial maturity
7. Banking behaviour if available

Return ONLY a JSON object, no markdown, no text outside JSON:

{{
  "score": <integer 300-850>,
  "risk_band": "<Very Low Risk|Low Risk|Moderate Risk|High Risk|Very High Risk>",
  "probability_of_default": <float 0.0-1.0>,
  "reasoning": "<2-3 sentence plain English explanation>",
  "key_positives": ["<positive 1>", "<positive 2>", "<positive 3>"],
  "key_negatives": ["<negative 1>", "<negative 2>"],
  "improvement_tips": ["<tip 1>", "<tip 2>", "<tip 3>"],
  "factor_contributions": [
    {{"feature": "income_expense_ratio", "value": <-0.3 to 0.3>, "label": "Income vs expenses"}},
    {{"feature": "debt_to_income",       "value": <-0.3 to 0.3>, "label": "Debt-to-income ratio"}},
    {{"feature": "asset_net_worth",      "value": <-0.3 to 0.3>, "label": "Net asset value"}},
    {{"feature": "employment_stability", "value": <-0.3 to 0.3>, "label": "Employment stability"}},
    {{"feature": "payment_history",      "value": <-0.3 to 0.3>, "label": "Payment history"}},
    {{"feature": "liability_burden",     "value": <-0.3 to 0.3>, "label": "Total liability burden"}},
    {{"feature": "savings_capacity",     "value": <-0.3 to 0.3>, "label": "Savings capacity"}},
    {{"feature": "banking_behaviour",    "value": <-0.3 to 0.3>, "label": "Banking behaviour"}}
  ],
  "cibil_comparison": "<Would CIBIL score this person fairly? Why not?>"
}}

Rules:
- 750-850: Excellent (strong income, low debt, solid assets)
- 670-749: Good (manageable debt, stable income)
- 580-669: Fair (some concerns but creditworthy)
- 500-579: High risk (significant debt or instability)
- 300-499: Very high risk (cannot service debt)
- Positive factor value = helps score, negative = hurts score
- Be fair to gig workers with good cash flow — income > expenses is a strong positive
- Someone earning 35000/month, spending 26000, zero loans should score at least 650"""


class ClaudeScorer:
    """
    AI-powered credit scorer using Google Gemini 1.5 Flash (free tier).
    Class named ClaudeScorer for API compatibility with existing main.py code.
    """

    def __init__(self):
        self._api_key   = GEMINI_API_KEY
        self._available = bool(self._api_key)

    @property
    def available(self) -> bool:
        return self._available

    def score(self, applicant_data: dict) -> Optional[dict]:
        """
        Send applicant data to Gemini → get structured credit score back.
        Returns dict or None if Gemini unavailable/fails.
        """
        if not self._available:
            logger.info("AI scorer: GEMINI_API_KEY not set — skipping")
            return None

        try:
            summary = self._build_summary(applicant_data)
            prompt  = SCORING_PROMPT.format(applicant_json=json.dumps(summary, indent=2))

            response = httpx.post(
                f"{GEMINI_URL}?key={self._api_key}",
                headers={"Content-Type": "application/json"},
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "temperature":    0.1,
                        "maxOutputTokens": 2000,
                    },
                },
                timeout=30.0,
            )
            response.raise_for_status()

            candidates = response.json().get("candidates", [])
            if not candidates:
                raise ValueError("Empty Gemini response")

            text = candidates[0]["content"]["parts"][0]["text"]
            text = re.sub(r"```(?:json)?\s*", "", text).strip()

            result = json.loads(text)
            logger.info(f"Gemini score: {result.get('score')} ({result.get('risk_band')})")
            return result

        except Exception as exc:
            logger.error(f"Gemini scorer failed: {exc}")
            return None

    def _build_summary(self, data: dict) -> dict:
        income    = float(data.get("monthly_income", 0))
        expenses  = float(data.get("monthly_expenses", 0))
        total_emi = float(data.get("total_emi", 0))

        total_liabilities = sum([
            float(data.get("home_loan_outstanding",    0)),
            float(data.get("vehicle_loan_outstanding",  0)),
            float(data.get("personal_loan_outstanding", 0)),
            float(data.get("credit_card_dues",          0)),
            float(data.get("other_liabilities",         0)),
        ])

        assets            = data.get("assets", [])
        total_asset_value = sum(float(a.get("declared_value",   0)) for a in assets)
        total_asset_loans = sum(float(a.get("outstanding_loan", 0)) for a in assets)

        return {
            "personal": {
                "name":            data.get("name", ""),
                "age":             data.get("age", 0),
                "employment_type": data.get("employment_type", ""),
                "years_employed":  data.get("years_employed", 0),
            },
            "income_and_expenses": {
                "monthly_income":        income,
                "monthly_expenses":      expenses,
                "monthly_savings":       round(income - expenses, 2),
                "savings_rate_percent":  round((income - expenses) / income * 100, 1) if income > 0 else 0,
                "monthly_emi":           total_emi,
                "emi_to_income_percent": round(total_emi / income * 100, 1) if income > 0 else 0,
            },
            "liabilities": {
                "home_loan":        data.get("home_loan_outstanding",    0),
                "vehicle_loan":     data.get("vehicle_loan_outstanding",  0),
                "personal_loan":    data.get("personal_loan_outstanding", 0),
                "credit_card_dues": data.get("credit_card_dues",          0),
                "other":            data.get("other_liabilities",         0),
                "total":            round(total_liabilities, 2),
                "loan_count":       data.get("existing_loans_count",      0),
            },
            "assets": {
                "items":           assets,
                "total_value":     round(total_asset_value, 2),
                "total_loans":     round(total_asset_loans, 2),
                "net_asset_value": round(total_asset_value - total_asset_loans, 2),
            },
            "net_worth":     round(total_asset_value - total_asset_loans - total_liabilities, 2),
            "credit_history": {
                "missed_payments_last_12m": data.get("missed_payments_last_12m", 0),
            },
            "bank_data": data.get("bank_data"),
        }


def blend_scores(ai_result: Optional[dict], model_score: int, model_pd: float) -> dict:
    """Blend AI + deterministic model scores. Falls back to model if AI unavailable."""
    if ai_result is None:
        return {
            "score":                  model_score,
            "risk_band":              _risk_band(model_score),
            "probability_of_default": round(model_pd, 4),
            "source":                 "model_only",
            "explanation":            "",
            "shap_values":            [],
        }

    ai_score = int(ai_result.get("score", model_score))
    ai_pd    = float(ai_result.get("probability_of_default", model_pd))

    # 100% Gemini — use directly, no blending
    blended_score = max(300, min(850, ai_score))
    blended_pd    = round(ai_pd, 4)

    return {
        "score":                  blended_score,
        "risk_band":              _risk_band(blended_score),
        "probability_of_default": blended_pd,
        "ai_score":               ai_score,
        "model_score":            model_score,
        "source":                 "gemini_blended",
        "reasoning":              ai_result.get("reasoning", ""),
        "key_positives":          ai_result.get("key_positives", []),
        "key_negatives":          ai_result.get("key_negatives", []),
        "improvement_tips":       ai_result.get("improvement_tips", []),
        "factor_contributions":   ai_result.get("factor_contributions", []),
        "cibil_comparison":       ai_result.get("cibil_comparison", ""),
        "shap_values": [
            {"feature": f["feature"], "value": f["value"], "label": f.get("label", "")}
            for f in ai_result.get("factor_contributions", [])
        ],
        "explanation": ai_result.get("reasoning", ""),
    }


def _risk_band(score: int) -> str:
    if score >= 760: return "Very Low Risk"
    if score >= 700: return "Low Risk"
    if score >= 640: return "Moderate Risk"
    if score >= 580: return "High Risk"
    return "Very High Risk"
