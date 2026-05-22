import json
import os
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

from core.json_utils import normalize_json_result, safe_json_loads


load_dotenv()


class UpstageService:
    def __init__(self):
        self.api_key = os.getenv("UPSTAGE_API_KEY")
        self.base_url = os.getenv("UPSTAGE_BASE_URL", "https://api.upstage.ai").rstrip("/")
        self.chat_model = os.getenv("UPSTAGE_CHAT_MODEL", "solar-pro3")
        self.embedding_model = os.getenv("UPSTAGE_EMBEDDING_MODEL", "embedding-query")
        self.timeout = httpx.Timeout(90.0)

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.2,
        response_format: dict | None = None,
    ) -> str:
        if not self.enabled:
            raise RuntimeError("UPSTAGE_API_KEY is not configured")

        payload: dict[str, Any] = {
            "model": model or self.chat_model,
            "messages": messages,
            "temperature": temperature,
        }
        if response_format:
            payload["response_format"] = response_format

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/v1/chat/completions",
                headers=self._headers(),
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not self.enabled:
            return []
        if not texts:
            return []

        payload = {"model": self.embedding_model, "input": texts}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/v1/embeddings",
                headers=self._headers(),
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            return [item["embedding"] for item in data.get("data", [])]

    async def ocr_or_parse_document(self, file_path: str) -> str:
        if not self.enabled:
            return f"[Document AI skipped: UPSTAGE_API_KEY is not configured] {Path(file_path).name}"

        path = Path(file_path)
        endpoint = f"{self.base_url}/v1/document-ai/document-parse"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        data = {"output_format": "markdown"}

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            with path.open("rb") as f:
                files = {"document": (path.name, f, "application/octet-stream")}
                response = await client.post(endpoint, headers=headers, files=files, data=data)
            if response.status_code >= 400:
                ocr_endpoint = f"{self.base_url}/v1/document-ai/ocr"
                with path.open("rb") as f:
                    files = {"document": (path.name, f, "application/octet-stream")}
                    response = await client.post(ocr_endpoint, headers=headers, files=files)
            response.raise_for_status()
            data = response.json()

        return (
            data.get("content", {}).get("text")
            or data.get("text")
            or data.get("html")
            or data.get("markdown")
            or json.dumps(data, ensure_ascii=False)
        )

    async def call_solar_json(
        self,
        system_prompt: str,
        user_payload: dict,
        temperature: float = 0.2,
        max_retries: int = 2,
    ) -> dict:
        if not self.enabled:
            return {
                "error": "UPSTAGE_API_KEY is not configured",
                "fallback": True,
                "agent_payload": user_payload,
            }

        messages = [
            {
                "role": "system",
                "content": (
                    system_prompt
                    + "\n\n출력 규칙: JSON만 출력한다. Markdown, 설명 문장, 코드블록을 사용하지 않는다."
                ),
            },
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ]

        last_text = ""
        for attempt in range(max_retries + 1):
            try:
                text = await self.chat(
                    messages=messages,
                    temperature=temperature if attempt == 0 else 0.0,
                    response_format={"type": "json_object"},
                )
                last_text = text
                return normalize_json_result(safe_json_loads(text))
            except Exception as exc:  # noqa: BLE001 - LLM/API repair path
                if attempt >= max_retries:
                    return {
                        "error": "Solar JSON output could not be parsed",
                        "fallback": True,
                        "detail": str(exc),
                        "raw_output": last_text[:2000],
                    }
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "이전 출력이 유효한 JSON이 아니었다. 같은 내용을 JSON 객체 또는 배열로만 "
                            "다시 작성하라. 코드블록과 설명 문장은 금지한다."
                        ),
                    }
                )
