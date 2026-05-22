# Divorce Camp Multi-Agent Backend

Hackathon MVP backend for document-grounded divorce reflection reports.

The backend is intentionally thin. Its main job is to run a Solar Pro 3 powered multi-agent pipeline:

```text
case info + uploaded files
  -> Upstage OCR / Document Parse
  -> Solar Pro 3 evidence extraction
  -> Upstage embeddings
  -> Pinecone case-evidence storage
  -> law.go.kr-based manual legal_basis.json
  -> Pinecone legal-knowledge storage
  -> Solar Pro 3 WifeLawyerAgent
  -> Solar Pro 3 HusbandLawyerAgent
  -> Solar Pro 3 JudgeAgent with Legal RAG
  -> Solar Pro 3 MediatorAgent
  -> final structured report
```

## Setup

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Fill `.env`:

```bash
UPSTAGE_API_KEY=...
PINECONE_API_KEY=...
PINECONE_INDEX_NAME=divorce-camp-agent
```

Run:

```bash
uvicorn main:app --reload
```

Health check:

```bash
curl http://127.0.0.1:8000/
```

## Endpoints

Create case:

```bash
curl -X POST http://127.0.0.1:8000/cases \
  -H "Content-Type: application/json" \
  -d '{
    "husband_name": "남편",
    "wife_name": "아내",
    "has_children": true,
    "children_count": 1,
    "marriage_duration": "7년",
    "main_conflict": "금전 문제와 양육 갈등",
    "divorce_intent": "undecided"
  }'
```

Upload files:

```bash
curl -X POST http://127.0.0.1:8000/cases/CASE_xxxxx/upload \
  -F "doc_type=chat_capture" \
  -F "files=@chat_1.png"
```

Initialize legal knowledge:

```bash
curl -X POST http://127.0.0.1:8000/legal/init
```

Reset legal knowledge namespace:

```bash
curl -X POST http://127.0.0.1:8000/legal/reset
```

Check Pinecone:

```bash
curl http://127.0.0.1:8000/pinecone/status
```

Search Legal RAG:

```bash
curl "http://127.0.0.1:8000/legal/search?q=상간녀%20위자료%20혼인파탄%20항변&top_k=5"
```

Run full pipeline:

```bash
curl -X POST http://127.0.0.1:8000/cases/CASE_xxxxx/run
```

Get report:

```bash
curl http://127.0.0.1:8000/cases/CASE_xxxxx/report
```

## Folder Structure

```text
backend/
  main.py
  services/
    upstage_service.py
    pinecone_service.py
    storage_service.py
  core/
    evidence_builder.py
    orchestrator.py
    schemas.py
    json_utils.py
  agents/
    wife_lawyer_agent.py
    husband_lawyer_agent.py
    judge_agent.py
    mediator_agent.py
  data/
    cases/
    legal_knowledge/legal_basis.json
```

## Agent Design

- `Evidence Builder`: Solar Pro 3 converts parsed document text into compact evidence items.
- `WifeLawyerAgent`: builds wife-side claims with evidence IDs and possible counterarguments.
- `HusbandLawyerAgent`: builds husband-side claims with evidence IDs and possible counterarguments.
- `JudgeAgent`: organizes legal issues using Legal RAG, without deciding who wins.
- `MediatorAgent`: gives practical next-step guidance based on divorce intent.

## Pinecone Namespaces

- `case-evidence`: user evidence vectors. Always filtered by `case_id`.
- `legal-knowledge`: law.go.kr-based manual legal seed vectors.

If Pinecone or Upstage keys are missing, the app returns fallback structures so local endpoint wiring can still be tested.

## Safety Disclaimer

본 결과는 AI 기반 정리 자료이며 실제 법률 자문이 아닙니다. 실제 판단과 대응은 변호사 또는 법률 전문가 상담이 필요합니다.

## Future Work

- frontend progress updates
- richer legal seed replacement with verified precedent metadata
- stronger evidence contradiction checking
- streaming agent status
- stricter production CORS and privacy controls
