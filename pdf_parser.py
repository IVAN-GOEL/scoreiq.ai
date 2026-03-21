# =============================================================================
# pdf_parser.py — Bank Statement PDF Parser (Claude AI Edition)
#
# Uses Claude claude-sonnet-4-20250514 to read bank statement PDFs and extract
# transactions with perfect accuracy across all Indian bank formats.
#
# Flow:
#   PDF bytes → base64 encode → Claude API (vision) → JSON transactions
#               → BankData features → scoring model
#
# Fallback chain:
#   Claude API → pdfplumber tables → pdfplumber text regex → empty BankData
#
# Setup:
#   Add ANTHROPIC_API_KEY to Railway environment variables
#   pip install anthropic pdfplumber
# =============================================================================

from __future__ import annotations

import base64
import io
import json
import os
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import httpx

try:
    import pdfplumber
    HAS_PDF = True
except ImportError:
    HAS_PDF = False

from schemas import BankData


# ── Keywords for fallback classification ──────────────────────────────────────
EMI_KEYWORDS    = ["EMI", "LOAN REPAY", "LOAN EMI", "HOME LOAN", "CAR LOAN", "PERSONAL LOAN"]
BOUNCE_KEYWORDS = ["BOUNCE", "RETURN", "DISHONOUR", "INSUFFICIENT", "INSUF FUNDS", "ECS RTN"]
SALARY_KEYWORDS = ["SALARY", "SAL CR", "PAYROLL", "NEFT-SALARY"]

DATE_FORMATS = [
    "%d-%m-%Y", "%d/%m/%Y", "%d-%m-%y", "%d/%m/%y",
    "%d %b %Y", "%d-%b-%Y", "%d/%b/%Y", "%d-%b-%y",
    "%Y-%m-%d", "%m/%d/%Y",
]

# ── Claude prompt ─────────────────────────────────────────────────────────────
CLAUDE_PROMPT = """You are a bank statement parser for Indian banks.

Extract ALL transactions from this bank statement PDF and return ONLY a JSON object.
No explanations, no markdown, no code blocks — just raw JSON.

Required format:
{
  "account_holder_name": "Full name from statement header",
  "bank_name": "Bank name",
  "statement_period": { "from": "YYYY-MM-DD", "to": "YYYY-MM-DD" },
  "opening_balance": 0.00,
  "closing_balance": 0.00,
  "transactions": [
    {
      "date": "YYYY-MM-DD",
      "narration": "Full transaction description",
      "type": "CREDIT or DEBIT",
      "amount": 0.00,
      "balance": 0.00
    }
  ]
}

Rules:
- Include EVERY transaction row, no skipping
- type must be exactly "CREDIT" or "DEBIT"
- amount must be a positive number
- date must be YYYY-MM-DD format
- narration should be the complete description from the statement"""


class BankStatementParser:
    """
    Parse Indian bank statement PDFs using Claude AI with pdfplumber fallback.

    Primary:   Claude Vision API  — handles any format, any bank, scanned PDFs
    Fallback1: pdfplumber tables  — structured digital PDFs
    Fallback2: pdfplumber + regex — non-tabular text PDFs
    Fallback3: empty BankData     — never crashes the API

    Public attributes after parse_pdf():
      extracted_name    str   — account holder name
      _short_statement  bool  — True if < 45 days of data
      raw_extraction    dict  — full Claude JSON (for /parse/pdf preview)
    """

    def __init__(self):
        self.extracted_name: str   = ""
        self._short_statement: bool = False
        self.raw_extraction: dict  = {}
        # 🔑 Set ANTHROPIC_API_KEY in Railway → Variables
        self._api_key = os.environ.get("ANTHROPIC_API_KEY", "")

    # =========================================================================
    # PUBLIC
    # =========================================================================

    def parse_pdf(self, pdf_bytes: bytes) -> BankData:
        """
        Parse a bank statement PDF (raw bytes) → BankData features.
        Tries Claude first, falls back to pdfplumber, never raises.
        """
        self.extracted_name   = ""
        self._short_statement = False
        self.raw_extraction   = {}

        # ── Claude (primary) ──────────────────────────────────────────────────
        if self._api_key:
            try:
                result = self._claude_extract(pdf_bytes)
                if result and result.get("transactions"):
                    self.raw_extraction = result
                    self.extracted_name = result.get("account_holder_name", "")
                    return self._compute_bank_data(
                        result["transactions"],
                        float(result.get("closing_balance", 0))
                    )
            except Exception as exc:
                print(f"[pdf] Claude failed ({exc}) — falling back to pdfplumber")

        # ── pdfplumber fallback ────────────────────────────────────────────────
        return self._pdfplumber_parse(pdf_bytes)

    # =========================================================================
    # CLAUDE EXTRACTION
    # =========================================================================

    def _claude_extract(self, pdf_bytes: bytes) -> dict:
        """
        Send PDF to Claude claude-sonnet-4-20250514 as a base64 document.
        Returns parsed JSON dict with transactions list.
        """
        pdf_b64 = base64.standard_b64encode(pdf_bytes).decode("utf-8")

        response = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key":         self._api_key,
                "anthropic-version": "2023-06-01",
                "content-type":      "application/json",
            },
            json={
                "model":      "claude-sonnet-4-20250514",
                "max_tokens": 4096,
                "messages": [{
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {
                                "type":       "base64",
                                "media_type": "application/pdf",
                                "data":       pdf_b64,
                            },
                        },
                        {"type": "text", "text": CLAUDE_PROMPT},
                    ],
                }],
            },
            timeout=60.0,
        )
        response.raise_for_status()

        # Extract text block from response
        text = next(
            (c["text"] for c in response.json()["content"] if c["type"] == "text"),
            ""
        )
        # Strip accidental markdown fences
        text = re.sub(r"```(?:json)?\s*", "", text).strip()

        result = json.loads(text)

        # Normalise all dates to YYYY-MM-DD
        for txn in result.get("transactions", []):
            txn["date"] = self._normalise_date(txn.get("date", ""))

        return result

    def _normalise_date(self, raw: str) -> str:
        """Convert any date string to YYYY-MM-DD."""
        for fmt in DATE_FORMATS + ["%Y-%m-%d"]:
            try:
                return datetime.strptime(raw.strip()[:len(fmt)], fmt).strftime("%Y-%m-%d")
            except Exception:
                continue
        return raw

    # =========================================================================
    # PDFPLUMBER FALLBACK
    # =========================================================================

    def _pdfplumber_parse(self, pdf_bytes: bytes) -> BankData:
        if not HAS_PDF:
            print("[pdf] pdfplumber not installed — returning empty BankData")
            return BankData()
        try:
            transactions = []
            closing      = 0.0
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                full_text  = ""
                all_tables = []
                for page in pdf.pages:
                    full_text += page.extract_text() or ""
                    tables = page.extract_tables()
                    if tables:
                        all_tables.extend(tables)
                self.extracted_name = self._name_from_text(full_text)
                if all_tables:
                    transactions = self._table_parse(all_tables)
                if not transactions:
                    transactions = self._text_parse(full_text)
                _, closing = self._extract_balances(full_text)
            return self._compute_bank_data(transactions, closing)
        except Exception as exc:
            print(f"[pdf] pdfplumber failed ({exc}) — returning empty BankData")
            return BankData()

    def _table_parse(self, tables) -> List[Dict]:
        out = []
        for table in tables:
            if not table or len(table) < 2:
                continue
            hdr    = [str(c).upper().strip() if c else "" for c in table[0]]
            d_col  = self._col(hdr, ["DATE", "TXN DATE", "VALUE DATE", "TRAN DATE"])
            n_col  = self._col(hdr, ["NARRATION", "DESCRIPTION", "PARTICULARS", "REMARKS"])
            dr_col = self._col(hdr, ["DEBIT", "DR", "WITHDRAWAL"])
            cr_col = self._col(hdr, ["CREDIT", "CR", "DEPOSIT"])
            b_col  = self._col(hdr, ["BALANCE", "BAL"])
            if d_col is None or n_col is None:
                continue
            for row in table[1:]:
                if not row or all(c is None or str(c).strip() == "" for c in row):
                    continue
                try:
                    narr = str(row[n_col]).strip() if n_col < len(row) else ""
                    if not narr or narr.upper() in ("-", "NONE", ""):
                        continue
                    dr  = self._amt(row[dr_col]) if dr_col is not None and dr_col < len(row) else 0.0
                    cr  = self._amt(row[cr_col]) if cr_col is not None and cr_col < len(row) else 0.0
                    typ = "CREDIT" if cr > 0 else "DEBIT" if dr > 0 else None
                    if typ is None:
                        continue
                    out.append({
                        "date":      str(row[d_col]).strip() if d_col < len(row) else "",
                        "narration": narr.upper(),
                        "amount":    cr if typ == "CREDIT" else dr,
                        "type":      typ,
                        "balance":   self._amt(row[b_col]) if b_col is not None and b_col < len(row) else 0.0,
                    })
                except Exception:
                    continue
        return out

    def _text_parse(self, text: str) -> List[Dict]:
        out      = []
        amt_re   = re.compile(r"(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)")
        date_re  = re.compile(r"\b(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})\b")
        for line in text.split("\n"):
            u = line.strip().upper()
            if len(u) < 10:
                continue
            dates   = date_re.findall(u)
            amounts = amt_re.findall(u)
            if not dates or not amounts:
                continue
            typ = None
            if "CR" in u.split() or any(k in u for k in ["CREDIT","SALARY","NEFT CR","IMPS CR"]):
                typ = "CREDIT"
            elif "DR" in u.split() or any(k in u for k in ["DEBIT","ATM","EMI","NEFT DR"]):
                typ = "DEBIT"
            if typ is None:
                continue
            amount = max([float(a.replace(",","")) for a in amounts], default=0.0)
            if amount < 1:
                continue
            out.append({"date": dates[0], "narration": u, "amount": amount, "type": typ, "balance": 0.0})
        return out

    def _name_from_text(self, text: str) -> str:
        skip = ["BANK","STATEMENT","ACCOUNT","BRANCH","IFSC","MICR","DATE",
                "PERIOD","PAGE","LIMITED","LTD","SAVINGS","CURRENT","JOINT"]
        for line in text.strip().split("\n")[:6]:
            line = line.strip()
            if not line or any(s in line.upper() for s in skip):
                continue
            words = line.split()
            if 2 <= len(words) <= 5 and all(w.replace(".","").replace("-","").isalpha() for w in words):
                return line.title()
        return ""

    def _extract_balances(self, text: str) -> Tuple[float, float]:
        u   = text.upper()
        m_o = re.search(r"OPENING BALANCE[:\s]+([0-9,]+\.?\d*)", u)
        m_c = re.search(r"CLOSING BALANCE[:\s]+([0-9,]+\.?\d*)", u)
        return (
            float(m_o.group(1).replace(",","")) if m_o else 0.0,
            float(m_c.group(1).replace(",","")) if m_c else 0.0,
        )

    # =========================================================================
    # FEATURE COMPUTATION (shared by both paths)
    # =========================================================================

    def _compute_bank_data(self, transactions: List[Dict], closing_balance: float) -> BankData:
        if not transactions:
            return BankData()

        months  = self._detect_months(transactions)
        credits = [t for t in transactions if str(t.get("type","")).upper() == "CREDIT"]
        debits  = [t for t in transactions if str(t.get("type","")).upper() == "DEBIT"]

        def a(t): return float(t.get("amount", 0))

        salary = [t for t in credits if a(t) > 5000
                  and any(k in str(t.get("narration","")).upper() for k in SALARY_KEYWORDS)]
        emis   = [t for t in debits
                  if any(k in str(t.get("narration","")).upper() for k in EMI_KEYWORDS)]
        bounces = [t for t in transactions
                   if any(k in str(t.get("narration","")).upper() for k in BOUNCE_KEYWORDS)]

        return BankData(
            avg_monthly_balance = closing_balance,
            avg_monthly_credits = sum(a(t) for t in credits) / months,
            avg_monthly_debits  = sum(a(t) for t in debits)  / months,
            emi_outflow         = sum(a(t) for t in emis)    / months,
            bounce_count_6m     = len(bounces),
            salary_detected     = len(salary) >= max(1, months * 0.7),
        )

    def _detect_months(self, transactions: List[Dict]) -> float:
        parsed = []
        for txn in transactions:
            raw = str(txn.get("date","")).strip()
            for fmt in DATE_FORMATS + ["%Y-%m-%d"]:
                try:
                    parsed.append(datetime.strptime(raw[:len(fmt)], fmt))
                    break
                except Exception:
                    continue
        if len(parsed) < 2:
            return 12.0
        days = (max(parsed) - min(parsed)).days
        if days < 15:
            self._short_statement = True
            return max(days / 30.0, 0.5)
        elif days < 45:
            self._short_statement = True
            return days / 30.0
        else:
            self._short_statement = False
            return max(days / 30.0, 1.0)

    def _col(self, header, keywords):
        for i, c in enumerate(header):
            if any(k in c for k in keywords):
                return i
        return None

    def _amt(self, val) -> float:
        if val is None: return 0.0
        cleaned = re.sub(r"[^\d.]", "", str(val).replace(",","").strip())
        try: return float(cleaned) if cleaned else 0.0
        except: return 0.0
