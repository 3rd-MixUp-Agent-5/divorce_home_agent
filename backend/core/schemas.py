from typing import Any, Literal

from pydantic import BaseModel, Field


DivorceIntent = Literal[
    "definitely_divorce",
    "undecided",
    "want_reconciliation",
    "need_partner_opinion",
]
DocType = Literal["card_statement", "receipt", "agreement_note", "chat_capture", "other"]
EvidenceType = Literal[
    "financial",
    "conversation",
    "agreement",
    "childcare",
    "emotional_context",
    "legal_issue",
    "other",
]
IssueTag = Literal[
    "property_division",
    "child_support",
    "custody",
    "parental_rights",
    "compensation",
    "financial_conflict",
    "agreement_note",
    "emotional_conflict",
    "divorce_ground",
    "reconciliation",
    "other",
]
RiskLevel = Literal["low", "medium", "high"]
Strength = Literal["weak", "medium", "strong"]


class CaseInfo(BaseModel):
    husband_name: str = "남편"
    wife_name: str = "아내"
    has_children: bool = False
    children_count: int = 0
    marriage_duration: str | None = None
    main_conflict: str | None = None
    divorce_intent: DivorceIntent = "undecided"


class UploadedFileInfo(BaseModel):
    file_id: str
    file_name: str
    doc_type: DocType = "other"
    path: str


class EvidenceItem(BaseModel):
    evidence_id: str
    case_id: str
    source_file_name: str
    doc_type: DocType = "other"
    evidence_type: EvidenceType = "other"
    party: str | None = None
    summary: str
    raw_quote: str = ""
    issue_tags: list[IssueTag] = Field(default_factory=list)
    risk_level: RiskLevel = "medium"
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class LegalKnowledgeItem(BaseModel):
    legal_basis_id: str
    source_type: str
    law_name: str | None = None
    article_number: str | None = None
    case_number: str | None = None
    court: str | None = None
    decision_date: str | None = None
    issue_type: str
    title: str
    summary: str
    content: str
    source: str
    source_url: str


class LawyerClaim(BaseModel):
    claim_id: str
    issue_type: str
    claim: str
    evidence_ids: list[str]
    reasoning: str
    strength: Strength = "medium"
    possible_counterargument: str


class LawyerAgentResult(BaseModel):
    agent_name: str
    summary: str
    claims: list[LawyerClaim] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)


class EvidenceValidationResult(BaseModel):
    valid_claims: list[dict[str, Any]] = Field(default_factory=list)
    weak_claims: list[dict[str, Any]] = Field(default_factory=list)
    invalid_claims: list[dict[str, Any]] = Field(default_factory=list)


class JudgeResult(BaseModel):
    agent_name: str = "JudgeAgent"
    summary: str
    issue_analysis: list[dict[str, Any]] = Field(default_factory=list)
    questions_for_lawyer: list[str] = Field(default_factory=list)
    legal_disclaimer: str


class MediatorResult(BaseModel):
    agent_name: str = "MediatorAgent"
    core_problem: str
    recommended_direction: str
    reasoning: str
    next_actions: list[str] = Field(default_factory=list)
    conversation_guidelines: list[str] = Field(default_factory=list)
    financial_guidelines: list[str] = Field(default_factory=list)
    childcare_guidelines: list[str] = Field(default_factory=list)
    warning_signs: list[str] = Field(default_factory=list)


class FinalReport(BaseModel):
    title: str = "AI 이혼숙려 리포트"
    case_summary: dict[str, Any]
    evidence_summary: dict[str, Any]
    wife_side: dict[str, Any]
    husband_side: dict[str, Any]
    evidence_validation: dict[str, Any]
    legal_issue_summary: dict[str, Any]
    mediator_advice: dict[str, Any]
    final_guideline: dict[str, Any]
    disclaimer: str
