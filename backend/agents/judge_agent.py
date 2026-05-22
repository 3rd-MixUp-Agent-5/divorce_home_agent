from services.upstage_service import UpstageService


LEGAL_DISCLAIMER = "본 결과는 AI 기반 정리 자료이며 실제 법률 자문이 아닙니다. 실제 판단과 대응은 변호사 또는 법률 전문가 상담이 필요합니다."


SYSTEM_PROMPT = f"""
너는 중립적인 법적 쟁점 정리 에이전트다.
역할:
- 실제 판결을 내리지 않는다.
- 양측 주장을 법적 쟁점별로 정리한다.
- 양측 변호사 agent의 토론과 MediatorAgent의 중재 내용을 참고하되, 법적 쟁점 정리는 독립적으로 한다.
- Legal RAG에서 제공된 legal_basis_id를 반드시 활용한다.
- 어느 쪽이 이긴다고 단정하지 않는다.
- 변호사 상담 전에 확인해야 할 질문을 만든다.
- 법률 자문이 아니라는 고지문을 반드시 포함한다.
- JSON으로만 출력한다.

금지:
- 승소/패소 단정
- 양육권 귀속 단정
- 위자료 가능성 단정
- 각서 효력 단정
- 실제 법률 상담처럼 말하기

출력 스키마:
{{
  "agent_name": "JudgeAgent",
  "summary": "string",
  "issue_analysis": [
    {{
      "issue_type": "property_division",
      "issue_title": "재산분할 쟁점",
      "wife_position": "string",
      "husband_position": "string",
      "related_claim_ids": [],
      "legal_basis_ids": [],
      "analysis": "string",
      "missing_evidence": [],
      "risk_level": "low | medium | high"
    }}
  ],
  "questions_for_lawyer": [],
  "legal_disclaimer": "{LEGAL_DISCLAIMER}"
}}
"""


async def run(
    case_info: dict,
    wife_result: dict,
    husband_result: dict,
    evidence_validation: dict,
    legal_context: list,
    agent_discussion: dict | None = None,
    mediator_result: dict | None = None,
    upstage_service: UpstageService | None = None,
) -> dict:
    upstage = upstage_service or UpstageService()
    result = await upstage.call_solar_json(
        system_prompt=SYSTEM_PROMPT,
        user_payload={
            "case_info": case_info,
            "wife_result": wife_result,
            "husband_result": husband_result,
            "evidence_validation": evidence_validation,
            "legal_context": legal_context,
            "agent_discussion": agent_discussion or {},
            "mediator_result": mediator_result or {},
            "task": "양측 주장, agent 토론, 중재 내용을 비교하여 승패가 아니라 법적 쟁점과 변호사 상담 질문을 최종 정리하라.",
        },
        temperature=0.15,
    )
    if result.get("fallback"):
        return {
            "agent_name": "JudgeAgent",
            "summary": "Solar 호출이 설정되지 않아 법적 쟁점 정리를 생성하지 못했습니다.",
            "issue_analysis": [],
            "questions_for_lawyer": ["UPSTAGE_API_KEY와 Pinecone legal-knowledge 초기화 여부를 확인하세요."],
            "legal_disclaimer": LEGAL_DISCLAIMER,
            "error": result.get("error"),
        }
    result.setdefault("agent_name", "JudgeAgent")
    result.setdefault("issue_analysis", [])
    result.setdefault("questions_for_lawyer", [])
    result["legal_disclaimer"] = result.get("legal_disclaimer") or LEGAL_DISCLAIMER
    return result
