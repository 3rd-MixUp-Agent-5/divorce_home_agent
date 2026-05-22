from services.upstage_service import UpstageService


SYSTEM_PROMPT = """
너는 아내 측 입장을 대변하는 AI 변호사 에이전트다.
규칙:
- 실제 변호사가 아니다.
- 법률 자문을 제공하지 않는다.
- evidence_items에 있는 내용만 사용한다.
- 모든 주장은 반드시 evidence_id를 포함해야 한다.
- 근거가 없으면 주장하지 않는다.
- 각 주장마다 상대방이 할 수 있는 반박을 possible_counterargument에 작성한다.
- 감정적 비난보다 증거 기반 주장을 우선한다.
- 아내 측에 유리한 주장을 구성하되, 불리한 약점도 숨기지 않는다.
- 확정적 법률 결론을 피하고, "쟁점이 될 가능성" 중심으로 작성한다.
- JSON으로만 출력한다.

출력 스키마:
{
  "agent_name": "WifeLawyerAgent",
  "summary": "string",
  "claims": [
    {
      "claim_id": "W_CLAIM_001",
      "issue_type": "child_support",
      "claim": "string",
      "evidence_ids": ["E001"],
      "reasoning": "string",
      "strength": "weak | medium | strong",
      "possible_counterargument": "string"
    }
  ],
  "weaknesses": [],
  "missing_evidence": []
}
"""


async def run(case_info: dict, evidence_items: list, upstage_service: UpstageService | None = None) -> dict:
    upstage = upstage_service or UpstageService()
    result = await upstage.call_solar_json(
        system_prompt=SYSTEM_PROMPT,
        user_payload={
            "case_info": case_info,
            "evidence_items": evidence_items,
            "task": "아내 측에 유리한 evidence-grounded claim을 만들고, 각 claim마다 가능한 반박도 작성하라.",
        },
        temperature=0.25,
    )
    if result.get("fallback"):
        return {
            "agent_name": "WifeLawyerAgent",
            "summary": "Solar 호출이 설정되지 않아 아내 측 주장을 생성하지 못했습니다.",
            "claims": [],
            "weaknesses": ["UPSTAGE_API_KEY 설정이 필요합니다."],
            "missing_evidence": [],
            "error": result.get("error"),
        }
    result.setdefault("agent_name", "WifeLawyerAgent")
    result.setdefault("claims", [])
    result.setdefault("weaknesses", [])
    result.setdefault("missing_evidence", [])
    return result


DISCUSSION_PROMPT = """
너는 아내 측 AI 변호사 에이전트다.
역할:
- 이미 생성된 아내 측 주장과 남편 측 주장을 비교한다.
- 남편 측 주장 중 반박할 부분, 일부 인정할 부분, 추가 증거가 필요한 부분을 구분한다.
- evidence_items에 있는 내용만 사용한다.
- 모든 반박 또는 보완 의견은 가능한 경우 evidence_id를 포함한다.
- 근거가 약한 부분은 약하다고 말한다.
- 감정적 비난보다 증거 기반 쟁점화를 우선한다.
- 실제 변호사가 아니며 법률 자문을 제공하지 않는다.
- JSON으로만 출력한다.

출력 스키마:
{
  "agent_name": "WifeLawyerAgent",
  "turn_type": "response_to_husband",
  "messages": [
    {
      "message_id": "W_TURN_001",
      "speaker": "아내 측 변호사",
      "content": "string",
      "responds_to_claim_ids": [],
      "evidence_ids": [],
      "stance": "rebuttal | concession | clarification | missing_evidence",
      "strength": "weak | medium | strong"
    }
  ],
  "summary": "string",
  "remaining_disputes": [],
  "missing_evidence": []
}
"""


async def respond_to_opponent(
    case_info: dict,
    evidence_items: list,
    wife_result: dict,
    husband_result: dict,
    upstage_service: UpstageService | None = None,
) -> dict:
    upstage = upstage_service or UpstageService()
    result = await upstage.call_solar_json(
        system_prompt=DISCUSSION_PROMPT,
        user_payload={
            "case_info": case_info,
            "evidence_items": evidence_items,
            "wife_result": wife_result,
            "husband_result": husband_result,
            "task": "남편 측 주장을 검토하고, 아내 측 입장에서 evidence-grounded response turn을 작성하라.",
        },
        temperature=0.25,
    )
    if result.get("fallback"):
        return {
            "agent_name": "WifeLawyerAgent",
            "turn_type": "response_to_husband",
            "messages": [],
            "summary": "Solar 호출이 설정되지 않아 아내 측 응답 turn을 생성하지 못했습니다.",
            "remaining_disputes": [],
            "missing_evidence": [],
            "error": result.get("error"),
        }
    result.setdefault("agent_name", "WifeLawyerAgent")
    result.setdefault("turn_type", "response_to_husband")
    result.setdefault("messages", [])
    result.setdefault("remaining_disputes", [])
    result.setdefault("missing_evidence", [])
    return result
