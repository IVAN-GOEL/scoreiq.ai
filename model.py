"""Credit scoring engine for ScoreIQ."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import math

from schemas import BankData, ScoreResponse, UserInputData


class CreditScoreEngine:
    """
    Lightweight deterministic engine.
    """

    def _income_to_expense_ratio(self, user: UserInputData) -> float:
        if user.monthly_expenses <= 0:
            return 5.0
        return user.monthly_income / user.monthly_expenses

    def _feature_contributions(
        self, user: UserInputData, bank: Optional[BankData]
    ) -> List[Tuple[str, float]]:
        ratio = self._income_to_expense_ratio(user)

        employment_bonus = {
            "salaried": 0.08,
            "self-employed": 0.03,
            "student": -0.05,
        }.get(user.employment_type.lower(), 0.0)

        features: List[Tuple[str, float]] = [
            ("income_to_expense_ratio", max(min((ratio - 1.0) * 0.15, 0.35), -0.25)),
            ("employment_stability", min(user.years_employed * 0.025, 0.20) + employment_bonus),
            ("missed_payments_12m", -min(user.missed_payments_last_12m * 0.08, 0.50)),
            ("existing_loans_count", -min(user.existing_loans_count * 0.04, 0.30)),
            ("age_profile", 0.03 if 24 <= user.age <= 50 else -0.02),
        ]

        if bank:
            cashflow_margin = bank.avg_monthly_credits - bank.avg_monthly_debits
            features.extend(
                [
                    ("avg_monthly_balance", min(bank.avg_monthly_balance / 250000, 0.12)),
                    ("cashflow_margin", max(min(cashflow_margin / 100000, 0.1), -0.12)),
                    ("emi_outflow", -max(min(bank.emi_outflow / max(user.monthly_income, 1), 0.18), 0)),
                    ("bounce_count_6m", -min(bank.bounce_count_6m * 0.035, 0.2)),
                    ("salary_detected", 0.05 if bank.salary_detected else -0.03),
                ]
            )

        return features

    def _score_mapping(self, pd: float) -> int:
        # Map PD (0..1) to 300..850 with wider spread.
        # Use power curve to amplify differences at extremes.
        # pd near 0 (very good) → score near 850
        # pd near 1 (very bad)  → score near 300
        import math
        # Stretch the score range using a non-linear mapping
        stretched = math.pow(pd, 0.7)   # power < 1 spreads low-risk scores higher
        raw = int(850 - stretched * 550)
        return max(300, min(850, raw))

    def _risk_band(self, score: int) -> str:
        if score >= 760:
            return "Very Low Risk"
        if score >= 700:
            return "Low Risk"
        if score >= 640:
            return "Moderate Risk"
        if score >= 580:
            return "High Risk"
        return "Very High Risk"

    def score(self, user_data: UserInputData, bank_data: Optional[BankData]) -> ScoreResponse:
        contributions = self._feature_contributions(user_data, bank_data)

        # Direct score calculation — sum contributions directly onto a 575 baseline
        # Range: contributions sum typically -0.8 to +0.8
        # Map to score: 575 + (sum * 350) → gives 300-850 range
        total_contribution = sum(v for _, v in contributions)
        score_value = int(575 + total_contribution * 350)
        score_value = max(300, min(850, score_value))

        # PD for display purposes only (inverse of score)
        pd = max(0.01, min(0.99, (850 - score_value) / 550))

        risk_band = self._risk_band(score_value)

        shap_values: List[Dict[str, float | str]] = [
            {"feature": name, "value": round(value, 4)} for name, value in contributions
        ]
        top_driver = max(contributions, key=lambda item: abs(item[1]))[0] if contributions else "n/a"
        explanation = (
            f"Estimated default probability is {pd:.2%}. "
            f"The strongest driver in this decision is '{top_driver}'."
        )

        return ScoreResponse(
            score=score_value,
            risk_band=risk_band,
            probability_of_default=round(pd, 4),
            shap_values=shap_values,
            explanation=explanation,
        )

    def get_fairness_metrics(self) -> Dict[str, object]:
        # Placeholder demo metrics for UI panel.
        return {
            "demographic_parity_gap": 0.036,
            "equal_opportunity_gap": 0.041,
            "group_wise_default_rate": {
                "group_a": 0.112,
                "group_b": 0.126,
            },
            "status": "monitor",
        }
