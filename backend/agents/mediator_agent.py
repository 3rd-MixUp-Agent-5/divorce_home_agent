from services.upstage_service import UpstageService


SYSTEM_PROMPT = """
너는 직설적이지만 균형 잡힌 제3자 조정자 에이전트다.
규칙:
- 한쪽 편만 들지 않는다.
- 사용자의 이혼 의사 여부를 반드시 반영한다.
- 법률 판단이 아니라 현실적인 다음 행동을 제시한다.
- 실제 인물의 말투나 캐릭터를 모방하지 않는다.
- TV 인물, 유명인, 특정 변호사 또는 판사의 화법을 따라 하지 않는다.
- 양측 변호사 agent의 토론을 읽고, 감정 주장과 증거 기반 쟁점을 분리한다.
- 감정적으로 공감하되, 행동 지침은 구체적으로 제시한다.
- 위험 신호가 있으면 안전과 전문가 도움을 우선 제안한다.
- JSON으로만 출력한다.

divorce_intent별 방향:
- definitely_divorce: evidence organization, lawyer consultation, child-related planning
- undecided: clarify key conflict, separate emotional/financial/legal issues, suggest counseling or mediated conversation
- want_reconciliation: conversation guide, financial transparency checklist, boundary setting
- need_partner_opinion: questions to ask spouse, safe conversation structure, what to document

출력 스키마:
{
  "agent_name": "MediatorAgent",
  "display_name": "서장훈 에이전트",
  "core_problem": "string",
  "recommended_direction": "divorce_preparation | reconciliation_attempt | legal_consultation | evidence_collection | undecided",
  "reasoning": "string",
  "debate_summary": "string",
  "points_to_calm_down": [],
  "balanced_reframe": [],
  "next_actions": [],
  "conversation_guidelines": [],
  "financial_guidelines": [],
  "childcare_guidelines": [],
  "warning_signs": []
}
"""


async def run(
    case_info: dict,
    wife_result: dict,
    husband_result: dict,
    agent_discussion: dict | None = None,
    upstage_service: UpstageService | None = None,
) -> dict:
    upstage = upstage_service or UpstageService()
    result = await upstage.call_solar_json(
        system_prompt=SYSTEM_PROMPT,
        user_payload={
            "case_info": case_info,
            "wife_result": wife_result,
            "husband_result": husband_result,
            "agent_discussion": agent_discussion or {},
            "task": "양측 변호사 agent의 주장과 토론을 바탕으로, 현실적인 중재 메시지와 다음 행동을 작성하라.",
        },
        temperature=0.3,
    )
    if result.get("fallback"):
        return {
            "agent_name": "MediatorAgent",
            "display_name": "서장훈 에이전트",
            "core_problem": case_info.get("main_conflict") or "핵심 갈등 정보가 부족합니다.",
            "recommended_direction": "evidence_collection",
            "reasoning": "Solar 호출이 설정되지 않아 구체적 조정 가이드를 생성하지 못했습니다.",
            "debate_summary": "",
            "points_to_calm_down": [],
            "balanced_reframe": [],
            "next_actions": ["업로드 문서와 case_info를 먼저 정리한 뒤 API key 설정 후 pipeline을 다시 실행하세요."],
            "conversation_guidelines": [],
            "financial_guidelines": [],
            "childcare_guidelines": [],
            "warning_signs": [],
            "error": result.get("error"),
        }
    result.setdefault("agent_name", "MediatorAgent")
    result.setdefault("display_name", "서장훈 에이전트")
    result.setdefault("debate_summary", "")
    result.setdefault("points_to_calm_down", [])
    result.setdefault("balanced_reframe", [])
    result.setdefault("next_actions", [])
    result.setdefault("conversation_guidelines", [])
    result.setdefault("financial_guidelines", [])
    result.setdefault("childcare_guidelines", [])
    result.setdefault("warning_signs", [])
    return result
