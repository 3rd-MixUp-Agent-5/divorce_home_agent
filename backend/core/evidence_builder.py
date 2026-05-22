import re
from pathlib import Path
from typing import Any

import pandas as pd

from services.upstage_service import UpstageService


ALLOWED_EVIDENCE_TYPES = {
    "financial",
    "conversation",
    "agreement",
    "childcare",
    "emotional_context",
    "legal_issue",
    "other",
}
ALLOWED_ISSUE_TAGS = {
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
}


SYSTEM_PROMPT = """
너는 이혼 관련 자료를 evidence item으로 변환하는 Evidence Builder다.
역할:
- 업로드된 문서 텍스트를 읽고, 이후 변호사 에이전트가 사용할 수 있는 evidence item으로 변환한다.
- 법적 판단을 하지 않는다.
- 감정적 해석을 과장하지 않는다.
- 원문 전체를 복사하지 말고 짧은 raw_quote만 남긴다.
- 개인정보는 최대한 마스킹한다.
- 출력은 JSON 배열 또는 {"items": [...]} 형태로만 작성한다.

문서 유형:
- 카드 명세서
- 영수증
- 각서
- 카카오톡/문자 캡처
- 기타

evidence_type은 다음 중 하나:
financial, conversation, agreement, childcare, emotional_context, legal_issue, other

issue_tags는 다음 중 선택:
property_division, child_support, custody, parental_rights, compensation, financial_conflict, agreement_note, emotional_conflict, divorce_ground, reconciliation, other

각 item은 다음 필드를 포함한다:
source_file_name, doc_type, evidence_type, party, summary, raw_quote, issue_tags, risk_level, confidence
"""


def mask_sensitive(text: str) -> str:
    if not text:
        return ""
    masked = re.sub(r"\b\d{6}-\d{7}\b", "******-*******", text)
    masked = re.sub(r"\b01[016789]-?\d{3,4}-?\d{4}\b", "010-****-****", masked)
    masked = re.sub(r"\b0\d{1,2}-?\d{3,4}-?\d{4}\b", "****-****-****", masked)
    masked = re.sub(r"\b\d{4}[- ]\d{4}[- ]\d{4}[- ]\d{4}\b", "****-****-****-****", masked)
    return masked


def _truncate(text: str, limit: int = 6000) -> str:
    text = mask_sensitive(text)
    return text if len(text) <= limit else text[:limit] + "\n...[truncated]"


def _dataframe_to_text(path: Path) -> str:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)

    head = df.head(80).fillna("").to_dict(orient="records")
    summary = {
        "columns": list(df.columns),
        "row_count": int(len(df)),
        "sample_rows": head,
    }
    return str(summary)


def _plain_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="cp949", errors="ignore")


async def _extract_file_text(file_info: dict, upstage_service: UpstageService) -> dict[str, Any]:
    path = Path(file_info["path"])
    suffix = path.suffix.lower()

    if suffix in {".xlsx", ".xls", ".csv"}:
        extracted = _dataframe_to_text(path)
    elif suffix in {".txt", ".md"}:
        extracted = _plain_text(path)
    elif suffix in {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}:
        extracted = await upstage_service.ocr_or_parse_document(str(path))
    else:
        extracted = f"Unsupported file type for direct parsing: {path.name}"

    return {
        "file_id": file_info.get("file_id"),
        "file_name": file_info.get("file_name") or path.name,
        "doc_type": file_info.get("doc_type", "other"),
        "extracted_text": _truncate(extracted),
    }


def _fallback_evidence(case_id: str, documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items = []
    for idx, doc in enumerate(documents, start=1):
        text = doc.get("extracted_text", "")
        items.append(
            {
                "evidence_id": f"E{idx:03d}",
                "case_id": case_id,
                "source_file_name": doc.get("file_name", ""),
                "doc_type": doc.get("doc_type", "other"),
                "evidence_type": "other",
                "party": None,
                "summary": f"{doc.get('file_name')} 문서에서 추출된 원문을 기반으로 evidence 생성이 필요합니다.",
                "raw_quote": mask_sensitive(text[:120]),
                "issue_tags": ["other"],
                "risk_level": "medium",
                "confidence": 0.3,
            }
        )
    return items


def _normalize_confidence(value: Any) -> float:
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))
    if isinstance(value, str):
        lowered = value.strip().lower()
        mapping = {
            "high": 0.85,
            "strong": 0.85,
            "medium": 0.6,
            "moderate": 0.6,
            "low": 0.35,
            "weak": 0.35,
        }
        if lowered in mapping:
            return mapping[lowered]
        try:
            return max(0.0, min(1.0, float(lowered)))
        except ValueError:
            return 0.5
    return 0.5


def _normalize_items(case_id: str, result: dict, documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    raw_items = result.get("items") if isinstance(result, dict) else result
    if not isinstance(raw_items, list) or not raw_items:
        return _fallback_evidence(case_id, documents)

    normalized = []
    for idx, item in enumerate(raw_items, start=1):
        if not isinstance(item, dict):
            continue
        issue_tags = [
            tag if tag in ALLOWED_ISSUE_TAGS else "other"
            for tag in item.get("issue_tags", ["other"])
            if isinstance(tag, str)
        ] or ["other"]
        evidence_type = item.get("evidence_type", "other")
        if evidence_type not in ALLOWED_EVIDENCE_TYPES:
            evidence_type = "other"
        risk_level = item.get("risk_level", "medium")
        if risk_level not in {"low", "medium", "high"}:
            risk_level = "medium"
        normalized.append(
            {
                "evidence_id": f"E{idx:03d}",
                "case_id": case_id,
                "source_file_name": item.get("source_file_name") or "",
                "doc_type": item.get("doc_type", "other"),
                "evidence_type": evidence_type,
                "party": item.get("party"),
                "summary": mask_sensitive(item.get("summary", "")),
                "raw_quote": mask_sensitive(item.get("raw_quote", ""))[:300],
                "issue_tags": issue_tags,
                "risk_level": risk_level,
                "confidence": _normalize_confidence(item.get("confidence", 0.5)),
            }
        )
    return normalized or _fallback_evidence(case_id, documents)


async def extract_documents(uploaded_files: list, upstage_service: UpstageService) -> list[dict[str, Any]]:
    documents = []
    for file_info in uploaded_files:
        documents.append(await _extract_file_text(file_info, upstage_service))
    return documents


async def build_evidence_from_documents(
    case_id: str,
    documents: list[dict[str, Any]],
    upstage_service: UpstageService,
) -> list:
    if not documents:
        return []

    result = await upstage_service.call_solar_json(
        system_prompt=SYSTEM_PROMPT,
        user_payload={
            "case_id": case_id,
            "documents": documents,
            "task": "각 문서에서 변호사 Agent가 사용할 수 있는 evidence item을 추출하라.",
        },
        temperature=0.15,
    )
    if result.get("fallback"):
        return _fallback_evidence(case_id, documents)
    return _normalize_items(case_id, result, documents)


async def build_evidence(case_id: str, uploaded_files: list, upstage_service: UpstageService) -> list:
    documents = await extract_documents(uploaded_files, upstage_service)
    return await build_evidence_from_documents(case_id, documents, upstage_service)
