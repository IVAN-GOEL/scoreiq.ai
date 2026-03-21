"""
ScoreIQ — Setu Account Aggregator Client

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 CREDENTIAL SETUP — 3 things to fill in
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Go to: https://bridge.setu.co
2. Login → Available Products → Data → Account Aggregator
3. Click "Add" → Create FIU app → fill in details
4. Under API Keys, copy the 3 values into the constants below
5. Set REDIRECT_URL to wherever your frontend lives
   (for local testing, use https://beeceptor.com to make a free mock endpoint)

Docs: https://docs.setu.co/data/account-aggregator/quickstart
"""

from __future__ import annotations
import os
import httpx
from datetime import datetime, timezone, timedelta
from uuid import uuid4
from typing import Dict, List

from schemas import BankData, ConsentResponse


SETU_CLIENT_ID           = os.environ.get("SETU_CLIENT_ID", "")
SETU_CLIENT_SECRET       = os.environ.get("SETU_CLIENT_SECRET", "")
SETU_PRODUCT_INSTANCE_ID = os.environ.get("SETU_PRODUCT_INSTANCE_ID", "")
REDIRECT_URL             = os.environ.get("REDIRECT_URL", "https://scoreiqver.vercel.app/consent/callback")
SETU_BASE_URL            = "https://fiu-sandbox.setu.co"

USE_MOCK = not bool(SETU_CLIENT_ID)



class SetuAAClient:
    def __init__(self) -> None:
        self._mock_consents: Dict[str, Dict] = {}
        self._headers = {
            "x-client-id": SETU_CLIENT_ID,
            "x-client-secret": SETU_CLIENT_SECRET,
            "x-product-instance-id": SETU_PRODUCT_INSTANCE_ID,
            "Content-Type": "application/json",
        }

    # ─────────────────────────────────────────
    # CONSENT
    # ─────────────────────────────────────────

    def create_consent(
        self,
        mobile: str,
        purpose: str,
        fi_types: List[str],
        date_range_months: int = 6,
    ) -> ConsentResponse:
        if USE_MOCK:
            return self._mock_create_consent(mobile)

        now = datetime.now(timezone.utc)
        from_date = (now - timedelta(days=date_range_months * 30)).strftime("%Y-%m-%dT%H:%M:%SZ")
        to_date   = now.strftime("%Y-%m-%dT%H:%M:%SZ")

        # Setu expects VUA format: 9999999999@setu
        clean = mobile.replace("+91", "").replace(" ", "").replace("-", "")
        vua   = f"{clean}@setu"

        payload = {
            "redirectUrl": REDIRECT_URL,
            "vua":         vua,
            "consentDuration": {"unit": "MONTH", "value": 1},
            "dataRange":   {"from": from_date, "to": to_date},
            "consentTypes": ["TRANSACTIONS", "SUMMARY", "PROFILE"],
            "fiTypes":     fi_types,
            "consentMode": "STORE",
            "fetchType":   "ONETIME",
            "Frequency":   {"unit": "MONTH", "value": 1},
            "DataLife":    {"unit": "MONTH", "value": 1},
            "purpose": {
                "code":    "101",
                "refUri":  "https://api.rebit.org.in/aa/purpose/101.xml",
                "text":    purpose,
                "Category": {"type": "Personal Finance"},
            },
        }

        resp = httpx.post(
            f"{SETU_BASE_URL}/consents",
            headers=self._headers,
            json=payload,
            timeout=15.0,
        )
        resp.raise_for_status()
        data = resp.json()

        return ConsentResponse(
            consent_id=data["id"],
            redirect_url=data["url"],   # open this in browser for user to approve
            status="PENDING_USER_ACTION",
        )

    def get_consent_status(self, consent_id: str) -> Dict:
        if USE_MOCK:
            return self._mock_get_status(consent_id)

        resp = httpx.get(
            f"{SETU_BASE_URL}/consents/{consent_id}",
            headers=self._headers,
            timeout=10.0,
        )
        resp.raise_for_status()
        data = resp.json()
        return {
            "consent_id":  consent_id,
            "status":      data.get("status", "UNKNOWN"),
            # Possible values: PENDING | APPROVED | REJECTED | EXPIRED | REVOKED
            "approved_at": data.get("approvedAt"),
        }

    # ─────────────────────────────────────────
    # DATA FETCH
    # ─────────────────────────────────────────

    def fetch_data(self, consent_id: str) -> Dict:
        if USE_MOCK:
            return self._mock_fetch_data(consent_id)

        now       = datetime.now(timezone.utc)
        from_date = (now - timedelta(days=180)).strftime("%Y-%m-%dT%H:%M:%SZ")
        to_date   = now.strftime("%Y-%m-%dT%H:%M:%SZ")

        # Step 1 — initiate data session
        sess = httpx.post(
            f"{SETU_BASE_URL}/sessions",
            headers=self._headers,
            json={"consentId": consent_id, "dataRange": {"from": from_date, "to": to_date}, "format": "json"},
            timeout=15.0,
        )
        sess.raise_for_status()
        session_id = sess.json()["id"]

        # Step 2 — poll (use webhooks in production for better UX)
        import time
        for _ in range(12):
            time.sleep(2)
            poll = httpx.get(f"{SETU_BASE_URL}/sessions/{session_id}", headers=self._headers, timeout=10.0)
            poll.raise_for_status()
            result = poll.json()
            if result.get("status") == "COMPLETED":
                return result
            if result.get("status") in ("FAILED", "EXPIRED"):
                raise RuntimeError(f"Session {result.get('status')}")

        raise TimeoutError("Setu data session timed out")

    # ─────────────────────────────────────────
    # PARSING — ReBIT FI schema → BankData
    # ─────────────────────────────────────────

    def parse_to_bank_data(self, raw_data: Dict) -> BankData:
        if USE_MOCK:
            summary = raw_data.get("summary", {})
        else:
            summary = self._parse_rebit(raw_data)

        return BankData(
            avg_monthly_balance=float(summary.get("avg_monthly_balance", 0)),
            avg_monthly_credits=float(summary.get("avg_monthly_credits", 0)),
            avg_monthly_debits=float(summary.get("avg_monthly_debits", 0)),
            emi_outflow=float(summary.get("emi_outflow", 0)),
            bounce_count_6m=int(summary.get("bounce_count_6m", 0)),
            salary_detected=bool(summary.get("salary_detected", False)),
        )

    def _parse_rebit(self, raw: Dict) -> Dict:
        """Parse real Setu FI data (ReBIT AA schema) into our BankData summary."""
        try:
            accounts     = raw.get("accounts", [])
            acct         = accounts[0] if accounts else {}
            txns         = acct.get("Transactions", {}).get("Transaction", [])
            summary_data = acct.get("Summary", {})
            months       = 6

            credits = [t for t in txns if t.get("type") == "CREDIT"]
            debits  = [t for t in txns if t.get("type") == "DEBIT"]

            salary_txns = [
                t for t in credits
                if float(t.get("amount", 0)) > 5000
                and any(k in t.get("narration", "").upper() for k in ["SALARY", "SAL", "NEFT", "IMPS"])
            ]
            emis = [
                t for t in debits
                if any(k in t.get("narration", "").upper() for k in ["EMI", "LOAN", "REPAY"])
            ]
            bounces = [
                t for t in txns
                if any(k in t.get("narration", "").upper() for k in ["BOUNCE", "RETURN", "DISHONOUR", "INSUF"])
            ]

            return {
                "avg_monthly_balance": float(summary_data.get("currentBalance", 0)),
                "avg_monthly_credits": sum(float(t.get("amount", 0)) for t in credits) / months,
                "avg_monthly_debits":  sum(float(t.get("amount", 0)) for t in debits)  / months,
                "emi_outflow":         sum(float(t.get("amount", 0)) for t in emis)    / months,
                "bounce_count_6m":     len(bounces),
                "salary_detected":     len(salary_txns) >= 3,
            }
        except Exception as e:
            print(f"[setu] parse warning: {e}")
            return {}

    # ─────────────────────────────────────────
    # MOCK FALLBACKS
    # ─────────────────────────────────────────

    def _mock_create_consent(self, mobile: str) -> ConsentResponse:
        cid = str(uuid4())
        self._mock_consents[cid] = {"mobile": mobile, "status": "APPROVED"}
        return ConsentResponse(
            consent_id=cid,
            redirect_url=f"https://mock.setu.dev/consent/{cid}",
            status="PENDING_USER_ACTION",
        )

    def _mock_get_status(self, consent_id: str) -> Dict:
        c = self._mock_consents.get(consent_id)
        return {"consent_id": consent_id, "status": c["status"] if c else "NOT_FOUND"}

    def _mock_fetch_data(self, consent_id: str) -> Dict:
        if consent_id not in self._mock_consents:
            raise ValueError("Consent not found")
        return {
            "consent_id": consent_id,
            "summary": {
                "avg_monthly_balance": 42000.0,
                "avg_monthly_credits": 72000.0,
                "avg_monthly_debits":  64000.0,
                "emi_outflow":         12000.0,
                "bounce_count_6m":     1,
                "salary_detected":     True,
            },
        }
