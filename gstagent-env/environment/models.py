"""
Pydantic data models for the GST Agent Environment.

Defines type-safe schemas for the entire observation/action/reward contract.
Every field is typed — no plain dicts allowed (contest spec requirement).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Invoice(BaseModel):
    """A single GST invoice record."""

    invoice_id: str = Field(..., description="Unique invoice identifier, e.g. INV-001")
    supplier_gstin: str = Field(..., description="15-character GSTIN of supplier")
    buyer_gstin: str = Field(..., description="15-character GSTIN of buyer")
    invoice_date: str = Field(..., description="Invoice date in YYYY-MM-DD format")
    taxable_amount: float = Field(..., ge=0, description="Taxable value in INR")
    cgst: float = Field(default=0.0, ge=0, description="Central GST amount")
    sgst: float = Field(default=0.0, ge=0, description="State GST amount")
    igst: float = Field(default=0.0, ge=0, description="Integrated GST amount")
    hsn_code: str = Field(..., description="4-8 digit HSN/SAC code")
    item_description: str = Field(default="", description="Description of goods/services")


class GSTObservation(BaseModel):
    """The observation returned to the agent at each step."""

    session_id: str = Field(default="", description="UUID for this episode session")
    task_id: str = Field(..., description="Current task identifier")
    purchase_register: list[Invoice] = Field(
        default_factory=list, description="Invoices from the business's books"
    )
    gstr2b_data: list[Invoice] = Field(
        default_factory=list, description="Invoices from GSTR-2B (government view)"
    )
    current_matches: list[dict] = Field(
        default_factory=list, description="Matches recorded so far"
    )
    unresolved_count: int = Field(default=0, description="Invoices not yet matched/flagged")
    step_number: int = Field(default=0, description="Current step in the episode")
    max_steps: int = Field(default=20, description="Maximum steps allowed")
    last_action_error: str | None = Field(
        default=None, description="Error from last action, if any"
    )


class GSTAction(BaseModel):
    """An action the agent can take."""

    action_type: str = Field(
        ...,
        description="One of: match_invoice, flag_mismatch, compute_itc, submit_report",
    )
    invoice_id: str | None = Field(default=None, description="Target invoice ID")
    reason: str | None = Field(default=None, description="Reason for flag/mismatch")
    payload: dict | None = Field(default=None, description="Additional data for the action")


class GSTReward(BaseModel):
    """Multi-signal reward returned on episode completion."""

    total: float = Field(..., ge=0.0, le=1.0, description="Final clamped score")
    itc_accuracy: float = Field(default=0.0, description="ITC amount accuracy score")
    recall_score: float = Field(default=0.0, description="Discrepancy recall score")
    action_correctness: float = Field(default=0.0, description="Action recommendation score")
    efficiency_bonus: float = Field(default=0.0, description="Bonus for fewer steps")
    hallucination_penalty: float = Field(default=0.0, description="Penalty for fake invoice IDs")


class ResetRequest(BaseModel):
    """Request body for POST /reset."""

    task_id: str = Field(..., description="Task to load: invoice_match, itc_audit, full_recon")


class StepRequest(BaseModel):
    """Request body for POST /step."""

    session_id: str = Field(..., description="Session UUID from /reset response")
    action: GSTAction = Field(..., description="The action to execute")
