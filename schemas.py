# =============================================================================
# schemas.py — Pydantic models for ScoreIQ API
# =============================================================================

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# =============================================================================
# SIMPLE SCORING SCHEMAS
# =============================================================================

class UserInputData(BaseModel):
    name:                     str   = Field(default="", example="Ramesh Kumar")
    age:                      int   = Field(..., ge=18, le=100, example=29)
    monthly_income:           float = Field(..., ge=0, example=35000)
    monthly_expenses:         float = Field(..., ge=0, example=26000)
    employment_type:          str   = Field(..., example="gig worker")
    years_employed:           float = Field(default=0.0, ge=0, example=1.5)
    missed_payments_last_12m: int   = Field(default=0, ge=0, le=60, example=0)
    existing_loans_count:     int   = Field(default=0, ge=0, le=20, example=0)
    # Liabilities
    home_loan_outstanding:     float = Field(default=0, ge=0)
    vehicle_loan_outstanding:  float = Field(default=0, ge=0)
    personal_loan_outstanding: float = Field(default=0, ge=0)
    credit_card_dues:          float = Field(default=0, ge=0)
    other_liabilities:         float = Field(default=0, ge=0)
    total_emi:                 float = Field(default=0, ge=0)


class BankData(BaseModel):
    avg_monthly_balance: float = Field(default=0, ge=0)
    avg_monthly_credits: float = Field(default=0, ge=0)
    avg_monthly_debits:  float = Field(default=0, ge=0)
    emi_outflow:         float = Field(default=0, ge=0)
    bounce_count_6m:     int   = Field(default=0, ge=0)
    salary_detected:     bool  = Field(default=False)


class ScoreResponse(BaseModel):
    score:                  int   = Field(..., ge=300, le=850)
    risk_band:              str
    probability_of_default: float = Field(..., ge=0, le=1)
    shap_values:            List[Dict[str, Any]]
    explanation:            str


# =============================================================================
# SETU AA SCHEMAS
# =============================================================================

class ConsentRequest(BaseModel):
    mobile: str = Field(..., min_length=10, max_length=15, example="9800112345")


class ConsentResponse(BaseModel):
    consent_id:   str
    redirect_url: str
    status:       str


class DataFetchResponse(BaseModel):
    success:   bool
    bank_data: Optional[BankData]       = None
    raw:       Optional[Dict[str, Any]] = None


# =============================================================================
# FULL PIPELINE SCHEMAS
# =============================================================================

class TransactionIn(BaseModel):
    date:      str = Field(..., example="2025-01-03")
    amount:    str = Field(..., example="45000.00")
    type:      str = Field(..., example="CREDIT")
    narration: str = Field(..., example="NEFT/SALARY/ACME CORP")
    balance:   str = Field(default="0", example="52000.00")


class AssetIn(BaseModel):
    type:             str   = Field(..., example="real_estate")
    declared_value:   float = Field(..., example=3500000)
    outstanding_loan: float = Field(default=0.0, example=2000000)
    description:      str   = Field(default="", example="2BHK flat")


# ScoreRequest MUST come after AssetIn
class ScoreRequest(BaseModel):
    user_data: UserInputData
    bank_data: Optional[BankData]      = None
    assets:    Optional[List[AssetIn]] = None


class AssessRequest(BaseModel):
    transactions:          List[TransactionIn] = Field(...)
    assets:                List[AssetIn]       = Field(default=[])
    loan_amount_requested: Optional[float]     = Field(default=0.0)


class FactorOut(BaseModel):
    feature:    str
    label:      str
    shap_value: float
    direction:  str
    impact_pts: int


class AssessResponse(BaseModel):
    score:                  int   = Field(..., ge=300, le=850)
    band:                   str
    color:                  str
    risk_band:              str
    default_probability:    float = Field(..., ge=0, le=1)
    probability_of_default: float = Field(..., ge=0, le=1)
    features:               Dict[str, Any]
    top_factors:            List[FactorOut]
    shap_values:            List[Dict[str, Any]]
    explanation:            str
