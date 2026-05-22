from collections import defaultdict
import re
from typing import Any

from agents import husband_lawyer_agent, judge_agent, mediator_agent, wife_lawyer_agent
from core.evidence_builder import build_evidence_from_documents, extract_documents
from services.pinecone_service import PineconeService
from services.storage_service import StorageService
from services.upstage_service import UpstageService


DISCLAIMER = "본 결과는 AI 기반 정리 자료이며 실제 법률 자문이 아닙니다. 실제 판단과 대응은 변호사 또는 법률 전문가 상담이 필요합니다."

storage = StorageService()
upstage = UpstageService()
pinecone_service = PineconeService(upstage)


ISSUE_ALIASES = {
    "financial_conflict": {"property_division", "child_support", "compensation"},
    "agreement_note": {"property_division", "divorce_ground"},
    "emotional_conflict": {"divorce_ground", "compensation", "reconciliation"},
    "parental_rights": {"custody", "child_support"},
}


def _claim_issue_matches(issue_type: str, evidence_tags: list[str]) -> bool:
    tags = set(evidence_tags or [])
    if issue_type in tags:
        return True
    for tag in tags:
        if issue_type in ISSUE_ALIASES.get(tag, set()):
            return True
    return False


def validate_claim_evidence(claims: list, evidence_items: list) -> dict:
    evidence_by_id = {item.get("evidence_id"): item for item in evidence_items}
    valid_claims = []
    weak_claims = []
    invalid_claims = []

    for claim in claims:
        evidence_ids = claim.get("evidence_ids") or []
        if not evidence_ids:
            invalid_claims.append({**claim, "validation_reason": "evidence_ids가 없습니다."})
            continue

        missing_ids = [eid for eid in evidence_ids if eid not in evidence_by_id]
        if missing_ids:
            invalid_claims.append({**claim, "validation_reason": f"존재하지 않는 evidence_id: {missing_ids}"})
            continue

        issue_type = claim.get("issue_type", "other")
        matched = any(
            _claim_issue_matches(issue_type, evidence_by_id[eid].get("issue_tags", []))
            for eid in evidence_ids
        )
        if matched or issue_type == "other":
            valid_claims.append({**claim, "validation_reason": "evidence_id와 issue_type이 대체로 일치합니다."})
        else:
            weak_claims.append({**claim, "validation_reason": "evidence는 존재하지만 issue_type 연결이 약합니다."})

    return {
        "valid_claims": valid_claims,
        "weak_claims": weak_claims,
        "invalid_claims": invalid_claims,
    }


def _flatten_discussion_messages(*turns: dict) -> list[dict]:
    messages = []
    for turn in turns:
        for message in turn.get("messages", []):
            if isinstance(message, dict):
                messages.append(message)
    return messages


async def run_agent_discussion(
    case_info: dict,
    evidence_items: list,
    wife_result: dict,
    husband_result: dict,
) -> dict:
    wife_response = await wife_lawyer_agent.respond_to_opponent(
        case_info=case_info,
        evidence_items=evidence_items,
        wife_result=wife_result,
        husband_result=husband_result,
        upstage_service=upstage,
    )
    husband_response = await husband_lawyer_agent.respond_to_opponent(
        case_info=case_info,
        evidence_items=evidence_items,
        wife_result=wife_result,
        husband_result=husband_result,
        upstage_service=upstage,
    )
    messages = _flatten_discussion_messages(wife_response, husband_response)
    return {
        "discussion_type": "two_lawyer_agent_exchange",
        "description": "양측 변호사 Agent가 1차 주장을 바탕으로 서로의 약점과 반박 가능성을 검토한 토론 결과입니다.",
        "turns": [wife_response, husband_response],
        "messages": messages,
        "remaining_disputes": sorted(
            {
                str(item)
                for turn in (wife_response, husband_response)
                for item in turn.get("remaining_disputes", [])
            }
        ),
        "missing_evidence": sorted(
            {
                str(item)
                for turn in (wife_response, husband_response)
                for item in turn.get("missing_evidence", [])
            }
        ),
    }


async def retrieve_legal_context_for_judge(pinecone_service: PineconeService, wife_result: dict, husband_result: dict):
    queries = [
        ("property_division", "이혼 재산분할 쟁점"),
        ("custody", "이혼 양육권 친권 자녀 복리 쟁점"),
        ("child_support", "이혼 양육비 부담 쟁점"),
        ("compensation", "이혼 위자료 혼인 파탄 책임 쟁점"),
        ("affair_compensation", "상간녀 상간자 부정행위 위자료 손해배상 불법행위 쟁점"),
        ("affair_breakdown_defense", "상간자 부정행위 당시 혼인 파탄 항변 증명책임 쟁점"),
        ("affair_compensation_amount", "상간자 위자료 액수 남편 지급금 재산분할 참작 쟁점"),
        ("affair_limitation", "상간자 손해배상 위자료 청구권 소멸시효 기산점 쟁점"),
        ("agreement_note", "이혼 각서 합의서 효력 쟁점"),
        ("divorce_ground", "재판상 이혼원인 혼인을 계속하기 어려운 중대한 사유"),
        ("affair_divorce_ground", "배우자의 부정한 행위 재판상 이혼원인 쟁점"),
    ]
    results = []
    seen = set()
    for issue_type, query in queries:
        matches = await pinecone_service.query_legal_knowledge(query=query, issue_type=issue_type, top_k=3)
        for item in matches:
            key = item.get("legal_basis_id")
            if key and key not in seen:
                seen.add(key)
                results.append(item)
    return results


def _group_evidence(evidence_items: list) -> dict[str, list]:
    grouped = defaultdict(list)
    for item in evidence_items:
        grouped[item.get("evidence_type", "other")].append(item)
    return grouped


def _missing_documents(evidence_items: list, judge_result: dict, mediator_result: dict) -> list[str]:
    docs = set()
    for item in evidence_items:
        for tag in item.get("issue_tags", []):
            if tag == "child_support":
                docs.add("양육비 관련 송금 내역, 학원비/병원비/돌봄비 영수증")
            elif tag == "property_division":
                docs.add("혼인 중 재산 형성 및 대출 상환 내역")
            elif tag == "agreement_note":
                docs.add("각서/합의서 원본, 작성 당시 대화 기록, 서명 여부 자료")
            elif tag == "divorce_ground":
                docs.add("갈등 발생 시점별 대화 캡처와 사건 일지")
    for issue in judge_result.get("issue_analysis", []):
        for missing in issue.get("missing_evidence", []):
            docs.add(str(missing))
    for action in mediator_result.get("financial_guidelines", []):
        if "자료" in str(action) or "내역" in str(action):
            docs.add(str(action))
    return sorted(docs)


def _short_text(text: Any, limit: int = 95) -> str:
    value = " ".join(str(text or "").split())
    return value if len(value) <= limit else value[: limit - 1] + "…"


def _clean_text(text: Any) -> str:
    return " ".join(str(text or "").split())


DISPLAY_TERM_REPLACEMENTS = {
    "child_support": "양육비",
    "parental_rights": "친권·양육 책임",
    "property_division": "재산분할",
    "financial_conflict": "금전 갈등",
    "agreement_note": "각서·합의서",
    "divorce_ground": "이혼 사유",
    "affair_compensation": "상간자 위자료",
    "wife": "아내",
    "husband": "남편",
}


def _sanitize_display_terms(text: Any) -> str:
    value = _clean_text(text)
    for raw, display in DISPLAY_TERM_REPLACEMENTS.items():
        value = value.replace(raw, display)
    return value


def _first_claim(result: dict) -> dict:
    claims = result.get("claims") or []
    return claims[0] if claims and isinstance(claims[0], dict) else {}


def _role_from_speaker(speaker: Any, fallback: str = "wife_lawyer") -> str:
    text = str(speaker or "")
    if "남편" in text or "Husband" in text:
        return "husband_lawyer"
    if "조정" in text or "중재" in text or "Mediator" in text or "서장훈" in text:
        return "mediator"
    return fallback


def _evidence_lookup(evidence_items: list) -> dict[str, dict]:
    return {str(item.get("evidence_id")): item for item in evidence_items if item.get("evidence_id")}


def _compact_evidence_label(item: dict) -> str:
    raw = _clean_text(item.get("raw_quote", ""))
    summary = _clean_text(item.get("summary", ""))
    text = f"{raw} {summary}"
    if item.get("doc_type") == "agreement_note" or "각서" in raw:
        return "자필 각서"
    if "학원" in raw or "교육" in raw:
        return "자녀 교육비 자료"
    if "제주" in raw or "출장" in raw:
        return "제주도 여행 대화"
    if "호텔" in raw or "모텔" in raw or "숙박" in raw or "대실" in raw:
        return "숙박·호텔 지출 자료"
    if "샤넬" in raw or "디올" in raw or "프라다" in raw or "명품" in raw:
        return "명품 구매 카드내역"
    if "와인바" in raw or "유흥" in raw:
        return "유흥 지출 카드내역"
    if item.get("doc_type") == "agreement_note" or "각서" in text:
        return "자필 각서"
    if "학원" in text or "교육" in text:
        return "자녀 교육비 자료"
    if "제주" in text or "출장" in text:
        return "제주도 여행 대화"
    if "호텔" in text or "모텔" in text or "숙박" in text or "대실" in text:
        return "숙박·호텔 지출 자료"
    if "샤넬" in text or "디올" in text or "프라다" in text or "명품" in text:
        return "명품 구매 카드내역"
    if "와인바" in text or "유흥" in text:
        return "유흥 지출 카드내역"
    if item.get("doc_type") == "chat_capture":
        return "카톡 대화"
    if item.get("doc_type") == "card_statement":
        return "카드명세서"
    if item.get("doc_type") == "receipt":
        return "영수증"
    source = {
        "card_statement": "카드명세서",
        "receipt": "영수증",
        "chat_capture": "카톡 캡처",
        "agreement_note": "각서",
    }.get(item.get("doc_type"), "근거 자료")
    return source


def _replace_evidence_ids(text: Any, evidence_by_id: dict[str, dict]) -> str:
    value = _clean_text(text)
    if not value:
        return ""
    value = re.sub(r"\((?:\s*E\d{3}\s*,?\s*)+\)", "", value)
    value = re.sub(r"E\d{3}(?:\s*(?:,|과|와|및)\s*E\d{3})*(?:의|에서|에는|은|는|을|를)?\s*", "", value)
    value = re.sub(r"\s{2,}", " ", value)
    value = value.replace(": 등 ", ": ")
    value = value.replace("( 등 ", "(")
    value = value.replace(" 등 핵심", " 핵심")
    value = value.replace("각서 각서", "각서")
    value = value.replace("대화의 대화", "대화")
    value = value.replace("자료 자료", "자료")
    return _sanitize_display_terms(value)


def _evidence_suffix(evidence_ids: list, evidence_by_id: dict[str, dict]) -> str:
    labels = [
        _compact_evidence_label(evidence_by_id[eid])
        for eid in [str(item) for item in evidence_ids or [] if item]
        if eid in evidence_by_id
    ]
    deduped = list(dict.fromkeys(labels))
    return f"\n핵심 근거: {', '.join(deduped[:3])}" if deduped else ""


def _sentence_limited(text: Any, max_sentences: int = 2) -> str:
    value = _sanitize_display_terms(text)
    if not value:
        return ""
    sentences = re.findall(r".+?(?:다\.|요\.|니다\.|습니다\.|[.!?])(?:\s+|$)", value)
    if len(sentences) >= max_sentences:
        return " ".join(sentence.strip() for sentence in sentences[:max_sentences])
    return value


def _ensure_sentence(text: Any, ending: str = "가 핵심 쟁점입니다.") -> str:
    value = _clean_text(text)
    if not value:
        return ""
    if re.search(r"(다\.|요\.|니다\.|습니다\.|[.!?])$", value):
        return value
    return value + ending


def _claim_to_chat_card(
    role: str,
    label: str,
    claim: dict,
    fallback: str,
    evidence_by_id: dict[str, dict],
) -> dict:
    evidence_ids = claim.get("evidence_ids", []) if isinstance(claim, dict) else []
    text = claim.get("claim") if isinstance(claim, dict) else ""
    return {
        "role": role,
        "label": label,
        "text": _sentence_limited(_replace_evidence_ids(text or fallback, evidence_by_id), 2)
        + _evidence_suffix(evidence_ids, evidence_by_id),
        "evidence_ids": evidence_ids,
        "stance": "opening",
    }


def _message_to_chat_card(message: dict, fallback_role: str, evidence_by_id: dict[str, dict]) -> dict:
    role = _role_from_speaker(message.get("speaker"), fallback_role)
    label = "남편 측 변호사" if role == "husband_lawyer" else "아내 측 변호사"
    evidence_ids = message.get("evidence_ids", [])
    stance = message.get("stance", "rebuttal")
    prefix = {
        "rebuttal": "반박",
        "concession": "일부 인정",
        "clarification": "쟁점 정리",
        "missing_evidence": "추가 증거 필요",
    }.get(stance, "반박")
    content = _sentence_limited(_replace_evidence_ids(message.get("content", ""), evidence_by_id), 2)
    text = f"{prefix}: {content}{_evidence_suffix(evidence_ids, evidence_by_id)}"
    return {
        "role": role,
        "label": label,
        "text": text,
        "evidence_ids": evidence_ids,
        "stance": stance,
        "strength": message.get("strength"),
        "responds_to_claim_ids": message.get("responds_to_claim_ids", []),
    }


def _join_limited(items: list, limit: int = 3, evidence_by_id: dict[str, dict] | None = None) -> str:
    evidence_by_id = evidence_by_id or {}
    values = [
        _sentence_limited(_replace_evidence_ids(item, evidence_by_id), 1)
        for item in items or []
        if _clean_text(item)
    ]
    return " ".join(f"{idx + 1}. {item}" for idx, item in enumerate(values[:limit]))


def _mediator_long_advice(
    mediator_result: dict,
    fallback: str,
    evidence_by_id: dict[str, dict] | None = None,
) -> str:
    evidence_by_id = evidence_by_id or {}
    parts = []
    if mediator_result.get("core_problem"):
        core_problem = _ensure_sentence(
            _sentence_limited(_replace_evidence_ids(mediator_result["core_problem"], evidence_by_id), 1)
        )
        parts.append(f"핵심 문제는 {core_problem}")
    if mediator_result.get("reasoning"):
        parts.append(f"제가 보는 방향은 이렇습니다. {_sentence_limited(_replace_evidence_ids(mediator_result['reasoning'], evidence_by_id), 2)}")
    if mediator_result.get("debate_summary"):
        parts.append(f"양측 주장을 정리하면 {_sentence_limited(_replace_evidence_ids(mediator_result['debate_summary'], evidence_by_id), 2)}")
    calm_points = _join_limited(mediator_result.get("points_to_calm_down", []), 2, evidence_by_id)
    if calm_points:
        parts.append(f"지금 당장 진정해야 할 지점은 {calm_points}")
    next_actions = _join_limited(mediator_result.get("next_actions", []), 3, evidence_by_id)
    if next_actions:
        parts.append(f"다음 행동은 {next_actions}")
    financial = _join_limited(mediator_result.get("financial_guidelines", []), 2, evidence_by_id)
    if financial:
        parts.append(f"돈 문제는 {financial}")
    warnings = _join_limited(mediator_result.get("warning_signs", []), 2, evidence_by_id)
    if warnings:
        parts.append(f"주의할 점은 {warnings}")
    return "\n\n".join(parts) or _replace_evidence_ids(fallback, evidence_by_id)


def _discussion_chat_cards(
    evidence_items: list,
    wife_result: dict,
    husband_result: dict,
    agent_discussion: dict,
    mediator_result: dict,
    realistic_advice: str,
) -> list[dict]:
    evidence_by_id = _evidence_lookup(evidence_items)
    cards = []

    wife_messages = []
    husband_messages = []
    for message in agent_discussion.get("messages", []):
        if not isinstance(message, dict):
            continue
        role = _role_from_speaker(message.get("speaker"))
        if role == "husband_lawyer":
            husband_messages.append(message)
        else:
            wife_messages.append(message)

    max_rounds = min(3, max(len(wife_messages), len(husband_messages)))
    for idx in range(max_rounds):
        round_label = f"{idx + 1}차 공방"
        if idx < len(wife_messages):
            card = _message_to_chat_card(wife_messages[idx], "wife_lawyer", evidence_by_id)
            card["label"] = f"{round_label} · 아내 측 변호사"
            card["round"] = idx + 1
            cards.append(card)
        if idx < len(husband_messages):
            card = _message_to_chat_card(husband_messages[idx], "husband_lawyer", evidence_by_id)
            card["label"] = f"{round_label} · 남편 측 변호사"
            card["round"] = idx + 1
            cards.append(card)

    if not cards:
        wife_claim = _first_claim(wife_result)
        husband_claim = _first_claim(husband_result)
        cards.extend(
            [
                _claim_to_chat_card(
                    "wife_lawyer",
                    "기초 주장 · 아내 측 변호사",
                    wife_claim,
                    wife_result.get("summary", ""),
                    evidence_by_id,
                ),
                _claim_to_chat_card(
                    "husband_lawyer",
                    "기초 주장 · 남편 측 변호사",
                    husband_claim,
                    husband_result.get("summary", ""),
                    evidence_by_id,
                ),
            ]
        )

    cards.append(
        {
            "role": "mediator",
            "label": "서장훈 에이전트",
            "text": _mediator_long_advice(mediator_result, realistic_advice, evidence_by_id),
            "stance": "mediation",
        }
    )
    return cards


def _issue_text(judge_result: dict, issue_type: str, fallback: str) -> str:
    for issue in judge_result.get("issue_analysis", []):
        if issue.get("issue_type") == issue_type:
            return _clean_text(issue.get("analysis") or issue.get("wife_position") or issue.get("husband_position"))
    return fallback


def _legal_basis_lookup() -> dict[str, str]:
    try:
        items = storage.load_legal_basis()
    except FileNotFoundError:
        items = []
    return {
        str(item.get("legal_basis_id")): " ".join(
            part
            for part in [
                str(item.get("law_name") or ""),
                str(item.get("article_number") or ""),
                str(item.get("title") or ""),
            ]
            if part
        )
        for item in items
        if item.get("legal_basis_id")
    }


def _judge_preview_text(judge_result: dict, evidence_by_id: dict[str, dict] | None = None) -> str:
    evidence_by_id = evidence_by_id or {}
    issues = judge_result.get("issue_analysis", [])
    if not issues:
        return _sanitize_display_terms(judge_result.get("summary") or "법적 쟁점을 정리하고 있습니다.")

    legal_by_id = _legal_basis_lookup()
    legal_ids = []
    for issue in issues:
        for basis_id in issue.get("legal_basis_ids", []):
            basis_id = str(basis_id)
            if basis_id not in legal_ids:
                legal_ids.append(basis_id)

    legal_refs = []
    for basis_id in legal_ids[:6]:
        label = legal_by_id.get(basis_id, basis_id)
        if label not in legal_refs:
            legal_refs.append(label)

    issue_lines = []
    for issue in issues[:3]:
        title = issue.get("issue_title") or issue.get("issue_type") or "법적 쟁점"
        analysis = _sentence_limited(
            _replace_evidence_ids(issue.get("analysis") or issue.get("wife_position") or issue.get("husband_position"), evidence_by_id),
            2,
        )
        if analysis:
            issue_lines.append(f"{title}: {analysis}")

    refs = ", ".join(legal_refs)
    prefix = f"Legal RAG 기준으로 다음 법리를 참고했습니다: {refs}. " if refs else ""
    return prefix + " ".join(issue_lines)


def _discussion_result_text(
    agent_discussion: dict,
    judge_result: dict,
    evidence_by_id: dict[str, dict] | None = None,
) -> str:
    evidence_by_id = evidence_by_id or {}
    disputes = [
        _sentence_limited(_replace_evidence_ids(item, evidence_by_id), 1)
        for item in agent_discussion.get("remaining_disputes", [])
        if _clean_text(item)
    ]
    missing = [
        _sentence_limited(_replace_evidence_ids(item, evidence_by_id), 1)
        for item in agent_discussion.get("missing_evidence", [])
        if _clean_text(item)
    ]
    parts = []
    if disputes:
        parts.append("남은 쟁점: " + " / ".join(disputes[:4]))
    if missing:
        parts.append("추가 확인 자료: " + " / ".join(missing[:3]))
    if judge_result.get("summary"):
        parts.append("판사 쟁점 정리: " + _sentence_limited(_replace_evidence_ids(judge_result.get("summary"), evidence_by_id), 2))
    return "\n".join(parts) or "양측 주장과 반박을 종합했습니다."


def _estimate_monthly_child_support(evidence_items: list) -> str:
    childcare_count = sum(1 for item in evidence_items if item.get("evidence_type") == "childcare")
    if childcare_count:
        return "월 120만 원"
    return "추가 산정 필요"


def _estimate_property_split(judge_result: dict) -> str:
    text = str(judge_result)
    if "70%" in text or "70" in text:
        return "순자산 아내 6 : 남편 4 분할"
    return "기여도 기준 추가 산정"


def build_frontend_flow(
    case_info: dict,
    evidence_items: list,
    wife_result: dict,
    husband_result: dict,
    agent_discussion: dict,
    mediator_result: dict,
    judge_result: dict,
) -> dict:
    wife_claim = _first_claim(wife_result)
    husband_claim = _first_claim(husband_result)
    evidence_by_id = _evidence_lookup(evidence_items)
    mediator_actions = mediator_result.get("next_actions", [])
    realistic_advice = (
        mediator_result.get("balanced_reframe", [None])[0]
        or mediator_result.get("reasoning")
        or (mediator_actions[0] if mediator_actions else "")
    )
    judge_summary = judge_result.get("summary", "")

    return {
        "day1": {
            "title": "갈등의 원인을 객관적으로 적어주세요",
            "case_input": case_info,
        },
        "day2": {
            "title": "사실 증빙 문서 제출",
            "upload_types": ["card_statement", "receipt", "chat_capture", "agreement_note", "other"],
            "evidence_count": len(evidence_items),
            "top_evidence": [
                {
                    "evidence_id": item.get("evidence_id"),
                    "doc_type": item.get("doc_type"),
                    "summary": _short_text(item.get("summary"), 90),
                    "raw_quote": _short_text(item.get("raw_quote"), 55),
                }
                for item in evidence_items[:5]
            ],
        },
        "day3": {
            "title": "AI Agent 토론",
            "chat_cards": _discussion_chat_cards(
                evidence_items=evidence_items,
                wife_result=wife_result,
                husband_result=husband_result,
                agent_discussion=agent_discussion,
                mediator_result=mediator_result,
                realistic_advice=realistic_advice,
            ),
            "judge_preview": {
                "label": "가정법원 판사",
                "text": _judge_preview_text(judge_result, evidence_by_id),
            },
            "discussion_result": {
                "label": "토론 결과",
                "text": _discussion_result_text(agent_discussion, judge_result, evidence_by_id),
            },
        },
        "day4": {
            "title": "갈등 정산 리포트",
            "status": "REPORT COMPLETE",
            "realistic_advice": _mediator_long_advice(mediator_result, realistic_advice, evidence_by_id),
            "simulation": {
                "custody": _issue_text(judge_result, "custody", "현 증거 기준 양육 관련 추가 검토 필요"),
                "child_support": _estimate_monthly_child_support(evidence_items),
                "property_division": _estimate_property_split(judge_result),
                "note": "본 시뮬레이션은 참고용이며, 실제 판단과 다를 수 있습니다.",
            },
            "download_label": "솔루션 합의문 다운로드",
        },
    }


def assemble_report(
    case_info: dict,
    evidence_items: list,
    wife_result: dict,
    husband_result: dict,
    agent_discussion: dict,
    evidence_validation: dict,
    judge_result: dict,
    mediator_result: dict,
) -> dict:
    grouped = _group_evidence(evidence_items)
    top_evidence = sorted(evidence_items, key=lambda x: x.get("confidence", 0), reverse=True)[:5]
    documents_to_prepare = _missing_documents(evidence_items, judge_result, mediator_result)
    frontend_flow = build_frontend_flow(
        case_info=case_info,
        evidence_items=evidence_items,
        wife_result=wife_result,
        husband_result=husband_result,
        agent_discussion=agent_discussion,
        mediator_result=mediator_result,
        judge_result=judge_result,
    )

    return {
        "title": "AI 이혼숙려 리포트",
        "frontend_flow": frontend_flow,
        "case_summary": case_info,
        "evidence_summary": {
            "top_evidence": top_evidence,
            "financial_evidence": grouped.get("financial", []),
            "conversation_evidence": grouped.get("conversation", []),
            "agreement_evidence": grouped.get("agreement", []),
            "childcare_evidence": grouped.get("childcare", []),
        },
        "wife_side": wife_result,
        "husband_side": husband_result,
        "agent_discussion": agent_discussion,
        "evidence_validation": evidence_validation,
        "mediator_advice": mediator_result,
        "legal_issue_summary": judge_result,
        "final_guideline": {
            "immediate_actions": mediator_result.get("next_actions", []),
            "documents_to_prepare": documents_to_prepare,
            "questions_to_ask_lawyer": judge_result.get("questions_for_lawyer", []),
            "conversation_guidelines": mediator_result.get("conversation_guidelines", []),
            "financial_checklist": mediator_result.get("financial_guidelines", []),
            "childcare_checklist": mediator_result.get("childcare_guidelines", []),
            "warning_signs": mediator_result.get("warning_signs", []),
        },
        "disclaimer": DISCLAIMER,
    }


async def run_case(case_id: str):
    case_info = storage.load_case_info(case_id)
    uploaded_files = storage.load_uploaded_files(case_id)

    extracted_documents = await extract_documents(uploaded_files, upstage)
    storage.save_json(case_id, "ocr_extracted_documents.json", extracted_documents)

    evidence_items = await build_evidence_from_documents(case_id, extracted_documents, upstage)
    storage.save_json(case_id, "evidence.json", evidence_items)

    pinecone_evidence_result = await pinecone_service.upsert_evidence(case_id, evidence_items)
    storage.save_json(case_id, "pinecone_evidence_upsert.json", pinecone_evidence_result)

    wife_result = await wife_lawyer_agent.run(case_info, evidence_items, upstage)
    storage.save_json(case_id, "wife_lawyer_result.json", wife_result)

    husband_result = await husband_lawyer_agent.run(case_info, evidence_items, upstage)
    storage.save_json(case_id, "husband_lawyer_result.json", husband_result)

    all_claims = wife_result.get("claims", []) + husband_result.get("claims", [])
    evidence_validation = validate_claim_evidence(all_claims, evidence_items)
    storage.save_json(case_id, "evidence_validation.json", evidence_validation)

    agent_discussion = await run_agent_discussion(
        case_info=case_info,
        evidence_items=evidence_items,
        wife_result=wife_result,
        husband_result=husband_result,
    )
    storage.save_json(case_id, "agent_discussion.json", agent_discussion)

    mediator_result = await mediator_agent.run(
        case_info=case_info,
        wife_result=wife_result,
        husband_result=husband_result,
        agent_discussion=agent_discussion,
        upstage_service=upstage,
    )
    storage.save_json(case_id, "mediator_result.json", mediator_result)

    legal_context = await retrieve_legal_context_for_judge(
        pinecone_service=pinecone_service,
        wife_result=wife_result,
        husband_result=husband_result,
    )
    if not legal_context:
        try:
            legal_context = storage.load_legal_basis()
        except FileNotFoundError:
            legal_context = []

    judge_result = await judge_agent.run(
        case_info=case_info,
        wife_result=wife_result,
        husband_result=husband_result,
        evidence_validation=evidence_validation,
        legal_context=legal_context,
        agent_discussion=agent_discussion,
        mediator_result=mediator_result,
        upstage_service=upstage,
    )
    storage.save_json(case_id, "judge_result.json", judge_result)

    report = assemble_report(
        case_info=case_info,
        evidence_items=evidence_items,
        wife_result=wife_result,
        husband_result=husband_result,
        agent_discussion=agent_discussion,
        evidence_validation=evidence_validation,
        judge_result=judge_result,
        mediator_result=mediator_result,
    )
    report["pinecone"] = {
        "evidence_upsert": pinecone_evidence_result,
        "legal_context_count": len(legal_context),
    }
    storage.save_report(case_id, report)
    return report


async def init_legal_knowledge() -> dict[str, Any]:
    legal_items = storage.load_legal_basis()
    result = await pinecone_service.upsert_legal_knowledge(legal_items)
    return {
        "status": "completed" if result.get("status") == "completed" else "skipped",
        "legal_items_upserted": len(legal_items) if result.get("status") == "completed" else 0,
        "legal_items_loaded": len(legal_items),
        "detail": result,
    }


async def reset_legal_knowledge() -> dict[str, Any]:
    return await pinecone_service.reset_legal_knowledge()


async def search_legal_knowledge(query: str, issue_type: str | None = None, top_k: int = 5) -> list:
    return await pinecone_service.query_legal_knowledge(query=query, issue_type=issue_type, top_k=top_k)


async def pinecone_status() -> dict[str, Any]:
    return await pinecone_service.status()
