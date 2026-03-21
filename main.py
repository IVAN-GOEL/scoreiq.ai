# =============================================================================
# main.py — ScoreIQ Backend API
#
# Unified FastAPI server combining:
#   • ScoreIQ simple scoring   (/score/form-only, /score, /score/pdf)
#   • Credit pipeline scoring  (/assess, /assess/demo)
#   • Setu AA integration      (/consent/*, /data/fetch/*)
#   • PDF statement parsing    (/parse/pdf)
#   • Explainability           (/explain, /fairness/metrics)
#
# Run locally:
#   pip install -r requirements.txt
#   python credit_pipeline.py --train   # only needed once
#   uvicorn main:app --reload --port 8000
#
# API docs: http://localhost:8000/docs
# =============================================================================

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import json
import io

from model import CreditScoreEngine
from claude_scorer import ClaudeScorer, blend_scores
from setu import SetuAAClient
from pdf_parser import BankStatementParser
from credit_pipeline import (
    ApplicantInput,
    Asset,
    assess_applicant,
    load_model,
    SAMPLE_TRANSACTIONS,
    SAMPLE_ASSETS,
)
from schemas import (
    UserInputData,
    ScoreRequest,
    ScoreResponse,
    ConsentRequest,
    ConsentResponse,
    DataFetchResponse,
    AssessRequest,
    AssessResponse,
)

# =============================================================================
# App initialisation
# =============================================================================

app = FastAPI(
    title="ScoreIQ API",
    description=(
        "Alternative credit scoring for thin-file borrowers.\n\n"
        "Two scoring modes:\n"
        "• **Simple** (`/score/*`) — fast, form-based, no ML training needed\n"
        "• **Full pipeline** (`/assess`) — RandomForest + SHAP + asset scoring"
    ),
    version="2.0.0",
)

# CORS — allow the Vercel frontend to talk to this Railway backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialise singletons at startup — not on every request
engine        = CreditScoreEngine()   # lightweight deterministic scorer
claude_scorer = ClaudeScorer()         # Claude AI enhanced scorer
setu   = SetuAAClient()        # Setu AA client (mock if no creds)
parser = BankStatementParser() # PDF statement parser

# Load/train the RandomForest model once at startup
# If data/model.pkl is missing, trains automatically on synthetic data (~10 sec)
try:
    rf_model = load_model()
except Exception as exc:
    import logging
    logging.warning(f"RF model unavailable: {exc}  — /assess will be disabled")
    rf_model = None


# =============================================================================
# HEALTH
# =============================================================================

@app.get("/", tags=["Health"])
def root() -> dict:
    """Health check — confirms the server is running."""
    return {
        "status":       "ok",
        "service":      "ScoreIQ API",
        "version":      "2.0.0",
        "model_loaded": rf_model is not None,
    }


# =============================================================================
# SIMPLE SCORING — form-only (no ML training required)
# Uses a deterministic logistic model in model.py
# =============================================================================


@app.post("/score/enhanced", response_model=ScoreResponse, tags=["Simple Scoring"])
def score_enhanced(request: ScoreRequest) -> ScoreResponse:
    """
    Enhanced scoring — combines deterministic model + Claude AI.
    Accepts all form fields, liabilities, assets and optional bank data.
    Claude AI analyses everything together like a senior credit analyst.
    Final score = Claude (60%) + deterministic model (40%).
    Falls back to deterministic model if Claude unavailable.
    """
    try:
        # Build full applicant dict with ALL inputs
        applicant_data             = request.user_data.dict()
        applicant_data["assets"]   = [a.dict() for a in request.assets] if request.assets else []
        applicant_data["bank_data"]= request.bank_data.dict() if request.bank_data else None

        # Try Gemini first
        gemini_result = claude_scorer.score(applicant_data)

        if gemini_result:
            # Gemini available — use its score 100%
            score   = max(300, min(850, int(gemini_result.get("score", 600))))
            pd      = float(gemini_result.get("probability_of_default", 0.4))
            band    = gemini_result.get("risk_band", _risk_band_str(score))

            pos  = gemini_result.get("key_positives", [])
            neg  = gemini_result.get("key_negatives", [])
            tips = gemini_result.get("improvement_tips", [])
            explanation = (
                gemini_result.get("reasoning", "") + " " +
                ("Key strengths: " + "; ".join(pos[:2]) + ". " if pos else "") +
                ("Concerns: " + "; ".join(neg[:1]) + "." if neg else "")
            ).strip()

            shap = [
                {"feature": f["feature"], "value": f["value"], "label": f.get("label","")}
                for f in gemini_result.get("factor_contributions", [])
            ]

        else:
            # Gemini unavailable — fall back to deterministic model
            model_result = engine.score(request.user_data, request.bank_data)
            score       = model_result.score
            pd          = model_result.probability_of_default
            band        = model_result.risk_band
            explanation = model_result.explanation
            shap        = model_result.shap_values

        return ScoreResponse(
            score                  = score,
            risk_band              = band,
            probability_of_default = round(pd, 4),
            shap_values            = shap,
            explanation            = explanation,
        )
    except Exception as exc:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc)) from exc

@app.post("/score/form-only", response_model=ScoreResponse, tags=["Simple Scoring"])
def score_form_only(user_data: UserInputData) -> ScoreResponse:
    """
    Score using only the self-reported form fields.
    No bank connection or PDF needed.
    Returns 300–850 score + SHAP-style feature contributions.
    """
    try:
        return engine.score(user_data, bank_data=None)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/score", response_model=ScoreResponse, tags=["Simple Scoring"])
def score(request: ScoreRequest) -> ScoreResponse:
    """
    Score using form data + structured bank features from Setu AA.
    bank_data is optional — omit it to score form-only.
    """
    try:
        return engine.score(request.user_data, request.bank_data)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/score/pdf", response_model=ScoreResponse, tags=["Simple Scoring"])
async def score_with_pdf(
    file:      UploadFile = File(...),
    user_data: str        = Form(...),
) -> ScoreResponse:
    """
    Score using form data + uploaded bank statement PDF.

    Multipart request:
      file       — the PDF file (max 10 MB)
      user_data  — JSON string of UserInputData fields

    The parser extracts transactions → BankData features → scoring model.
    Falls back to form-only scoring if the PDF cannot be parsed.
    Also auto-fills the applicant name from the PDF header.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    try:
        user = UserInputData(**json.loads(user_data))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid user_data: {exc}") from exc

    bank_data = None
    try:
        pdf_bytes = await file.read()
        bank_data = parser.parse_pdf(pdf_bytes)
        if parser.extracted_name and not user.name:
            user.name = parser.extracted_name
    except Exception as exc:
        print(f"[pdf] parse failed: {exc}")   # non-fatal

    try:
        return engine.score(user, bank_data)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# =============================================================================
# PDF PARSING — preview endpoint (called before scoring)
# =============================================================================

@app.post("/parse/pdf", tags=["PDF"])
async def parse_pdf_only(file: UploadFile = File(...)) -> dict:
    """
    Parse a bank statement PDF and return extracted features + name.

    Called by the frontend immediately after the user selects a file to:
      1. Auto-fill the name field (with green flash animation)
      2. Warn the user if the statement is < 45 days (short_statement flag)
      3. Preview what will be used for scoring
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")
    try:
        pdf_bytes = await file.read()
        bank_data = parser.parse_pdf(pdf_bytes)
        return {
            "success":         True,
            "bank_data":       bank_data.dict(),
            "extracted_name":  parser.extracted_name or None,
            "short_statement": getattr(parser, "_short_statement", False),
        }
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not parse PDF: {exc}") from exc


# =============================================================================
# FULL PIPELINE SCORING — RandomForest + SHAP + asset scoring
# =============================================================================

@app.post("/assess", response_model=AssessResponse, tags=["Full Pipeline"])
def assess(body: AssessRequest) -> AssessResponse:
    """
    Full ML pipeline assessment.

    Accepts raw bank transactions + self-declared assets.
    Returns credit score + SHAP explanation of top 5 drivers.

    Transaction format (list of):
      { date, amount, type (CREDIT/DEBIT), narration, balance }

    Asset type options:
      real_estate | gold | fd | mutual_fund | epf_ppf | vehicle
    """
    if rf_model is None:
        raise HTTPException(status_code=503, detail="ML model not loaded. Try /score/form-only instead.")

    try:
        assets = [
            Asset(
                type=a.type,
                declared_value=a.declared_value,
                outstanding_loan=a.outstanding_loan,
                description=a.description,
            )
            for a in body.assets
        ]
        transactions = [t.model_dump() for t in body.transactions]
        applicant    = ApplicantInput(
            transactions=transactions,
            assets=assets,
            loan_amount_requested=body.loan_amount_requested or 0.0,
        )
        result = assess_applicant(applicant, rf_model)
        return AssessResponse(**result)
    except Exception as exc:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/assess/demo", response_model=AssessResponse, tags=["Full Pipeline"])
def assess_demo(profile: str = "salaried") -> AssessResponse:
    """
    Demo endpoint — returns a pre-baked assessment without live data.
    Perfect for hackathon demos when Setu connection isn't available.

    profile options: salaried | gig | struggling
    """
    if rf_model is None:
        raise HTTPException(status_code=503, detail="ML model not loaded.")

    GIG_TRANSACTIONS = [
        {"date": "2024-10-03", "amount": "42000", "type": "CREDIT",
         "narration": "UPI/FREELANCE/CLIENT A", "balance": "44000"},
        {"date": "2024-10-18", "amount": "3200",  "type": "DEBIT",
         "narration": "UPI/RENT", "balance": "40800"},
        {"date": "2024-11-10", "amount": "3200",  "type": "DEBIT",
         "narration": "UPI/RENT", "balance": "37600"},
        {"date": "2024-11-22", "amount": "8000",  "type": "DEBIT",
         "narration": "EMI/VEHICLE LOAN", "balance": "29600"},
        {"date": "2024-12-05", "amount": "9000",  "type": "CREDIT",
         "narration": "UPI/FREELANCE/CLIENT B", "balance": "37700"},
        {"date": "2024-12-10", "amount": "8000",  "type": "DEBIT",
         "narration": "EMI/VEHICLE LOAN", "balance": "29700"},
    ]

    STRUGGLING_TRANSACTIONS = [
        {"date": "2024-10-05", "amount": "12000", "type": "CREDIT",
         "narration": "NEFT/SALARY", "balance": "13000"},
        {"date": "2024-10-08", "amount": "6000",  "type": "DEBIT",
         "narration": "EMI/LOAN", "balance": "7000"},
        {"date": "2024-10-20", "amount": "6800",  "type": "DEBIT",
         "narration": "UPI/RENT", "balance": "200"},
        {"date": "2024-11-05", "amount": "12000", "type": "CREDIT",
         "narration": "NEFT/SALARY", "balance": "12200"},
        {"date": "2024-11-22", "amount": "6800",  "type": "DEBIT",
         "narration": "UPI/RENT", "balance": "-600"},
    ]

    profiles = {
        "salaried":   (SAMPLE_TRANSACTIONS, SAMPLE_ASSETS),
        "gig":        (GIG_TRANSACTIONS,    [Asset("vehicle", 400000, 320000)]),
        "struggling": (STRUGGLING_TRANSACTIONS, []),
    }

    if profile not in profiles:
        raise HTTPException(status_code=400, detail=f"Unknown profile. Choose: {list(profiles)}")

    txns, assets = profiles[profile]
    applicant    = ApplicantInput(transactions=txns, assets=assets)

    try:
        result = assess_applicant(applicant, rf_model)
        return AssessResponse(**result)
    except Exception as exc:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# =============================================================================
# SETU AA — consent flow + data fetch
# =============================================================================

@app.post("/consent/create", response_model=ConsentResponse, tags=["Setu AA"])
def create_consent(request: ConsentRequest) -> ConsentResponse:
    """
    Create a Setu AA consent request.
    Returns a redirect_url — open this URL for the user to approve in their bank app.
    Runs in mock mode automatically if SETU_CLIENT_ID env var is not set.
    """
    try:
        return setu.create_consent(
            mobile=request.mobile,
            purpose="Credit Score Assessment by ScoreIQ",
            fi_types=["DEPOSIT", "RECURRING_DEPOSIT"],
            date_range_months=6,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Setu AA error: {str(exc)}") from exc


@app.get("/consent/{consent_id}/status", tags=["Setu AA"])
def get_consent_status(consent_id: str) -> dict:
    """
    Poll consent approval status.
    Possible values: PENDING | APPROVED | REJECTED | EXPIRED | REVOKED
    """
    try:
        return setu.get_consent_status(consent_id)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/data/fetch/{consent_id}", response_model=DataFetchResponse, tags=["Setu AA"])
def fetch_financial_data(consent_id: str) -> DataFetchResponse:
    """
    Fetch bank transaction data after user has approved consent.
    Parses the Setu FI response into structured BankData for scoring.
    """
    try:
        raw_data = setu.fetch_data(consent_id)
        parsed   = setu.parse_to_bank_data(raw_data)
        return DataFetchResponse(success=True, bank_data=parsed)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/notifications", tags=["Setu AA"])
async def setu_notifications(request: Request) -> dict:
    """
    Setu AA webhook endpoint.
    Register this URL as the Test Callback URL in bridge.setu.co.
    Setu POSTs here when a consent status changes.
    """
    body = await request.json()
    print(f"[setu webhook] {body}")
    return {"status": "ok"}


# =============================================================================
# EXPLAINABILITY
# =============================================================================

@app.post("/explain", tags=["Explainability"])
def explain(request: ScoreRequest) -> dict:
    """Return top positive and negative SHAP factors for a given input."""
    try:
        result = engine.score(request.user_data, request.bank_data)
        pos = sorted([f for f in result.shap_values if f["value"] > 0],
                     key=lambda x: x["value"], reverse=True)[:3]
        neg = sorted([f for f in result.shap_values if f["value"] < 0],
                     key=lambda x: x["value"])[:3]
        return {
            "score":                result.score,
            "shap_values":          result.shap_values,
            "top_positive_factors": pos,
            "top_negative_factors": neg,
            "explanation":          result.explanation,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/fairness/metrics", tags=["Explainability"])
def fairness_metrics() -> dict:
    """
    Return model fairness audit metrics.
    Demonstrates the ScoreIQ advantage over traditional CIBIL for thin-file users.
    """
    return engine.get_fairness_metrics()


# =============================================================================
# ENTRYPOINT
# =============================================================================

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
