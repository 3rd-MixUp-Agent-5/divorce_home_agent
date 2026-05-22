import hashlib
import os
import time
from typing import Any

from dotenv import load_dotenv

from services.upstage_service import UpstageService


load_dotenv()


class PineconeService:
    def __init__(self, upstage_service: UpstageService | None = None):
        self.api_key = os.getenv("PINECONE_API_KEY")
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "divorce-camp-agent")
        self.cloud = os.getenv("PINECONE_CLOUD", "aws")
        self.region = os.getenv("PINECONE_REGION", "us-east-1")
        self.upstage = upstage_service or UpstageService()
        self.pc = None
        self.index = None
        self.last_error: str | None = None

    @property
    def enabled(self) -> bool:
        return self.index is not None and self.upstage.enabled

    def _client(self):
        if not self.api_key:
            self.last_error = "PINECONE_API_KEY is not configured"
            return None
        if self.pc is None:
            try:
                from pinecone import Pinecone

                self.pc = Pinecone(api_key=self.api_key)
            except Exception as exc:  # noqa: BLE001
                self.last_error = f"Pinecone client init failed: {exc}"
                return None
        return self.pc

    def _connect_index(self):
        pc = self._client()
        if pc is None:
            return None
        try:
            self.index = pc.Index(self.index_name)
            self.last_error = None
            return self.index
        except Exception as exc:  # noqa: BLE001
            self.index = None
            self.last_error = f"Pinecone index connection failed: {exc}"
            return None

    def _ensure_index(self, dimension: int):
        pc = self._client()
        if pc is None:
            return None
        try:
            if not pc.has_index(self.index_name):
                from pinecone import ServerlessSpec

                pc.create_index(
                    name=self.index_name,
                    dimension=dimension,
                    metric="cosine",
                    spec=ServerlessSpec(cloud=self.cloud, region=self.region),
                    timeout=60,
                )
                # Pinecone may need a short moment before the host resolves.
                for _ in range(10):
                    if pc.has_index(self.index_name):
                        break
                    time.sleep(1)
            return self._connect_index()
        except Exception as exc:  # noqa: BLE001
            self.index = None
            self.last_error = f"Pinecone index ensure failed: {exc}"
            return None

    def _index_dimension(self, index) -> int | None:
        try:
            stats = index.describe_index_stats()
            data = stats.to_dict() if hasattr(stats, "to_dict") else stats
            return data.get("dimension")
        except Exception:
            return None

    def _fit_embedding_dimension(self, embedding: list[float], target_dimension: int | None) -> tuple[list[float], bool]:
        if not target_dimension or len(embedding) == target_dimension:
            return embedding, False
        if len(embedding) > target_dimension:
            return embedding[:target_dimension], True
        return embedding + [0.0] * (target_dimension - len(embedding)), True

    def _chunk_text(self, text: str, chunk_size: int = 1200, overlap: int = 160) -> list[str]:
        clean = " ".join((text or "").split())
        if not clean:
            return []
        chunks = []
        start = 0
        while start < len(clean):
            end = min(start + chunk_size, len(clean))
            chunks.append(clean[start:end])
            if end >= len(clean):
                break
            start = max(0, end - overlap)
        return chunks

    async def status(self) -> dict[str, Any]:
        status: dict[str, Any] = {
            "pinecone_api_key_configured": bool(self.api_key),
            "upstage_api_key_configured": self.upstage.enabled,
            "index_name": self.index_name,
            "cloud": self.cloud,
            "region": self.region,
            "index_connected": False,
            "stats": None,
            "last_error": self.last_error,
        }
        index = self.index or self._connect_index()
        if index is None:
            status["last_error"] = self.last_error
            return status
        try:
            stats = index.describe_index_stats()
            status["index_connected"] = True
            status["stats"] = stats.to_dict() if hasattr(stats, "to_dict") else stats
            status["last_error"] = None
        except Exception as exc:  # noqa: BLE001
            status["last_error"] = f"Pinecone stats failed: {exc}"
        return status

    def _stable_id(self, prefix: str, value: str) -> str:
        digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]
        return f"{prefix}_{digest}"

    async def upsert_evidence(self, case_id: str, evidence_items: list):
        if not evidence_items:
            return {"status": "skipped", "count": 0, "reason": "No evidence items"}
        if not self.upstage.enabled:
            return {"status": "skipped", "count": 0, "reason": "UPSTAGE_API_KEY is not configured"}
        if not self.api_key:
            return {"status": "skipped", "count": 0, "reason": "PINECONE_API_KEY is not configured"}

        texts = [
            f"{item.get('summary', '')}\n{item.get('raw_quote', '')}\n{','.join(item.get('issue_tags', []))}"
            for item in evidence_items
        ]
        embeddings = await self.upstage.embed(texts)
        if not embeddings:
            return {"status": "failed", "count": 0, "reason": "Upstage embedding returned no vectors"}
        index = self._ensure_index(len(embeddings[0]))
        if index is None:
            return {"status": "failed", "count": 0, "reason": self.last_error}
        target_dimension = self._index_dimension(index)
        dimension_adjusted = bool(target_dimension and target_dimension != len(embeddings[0]))

        vectors = []
        for item, embedding in zip(evidence_items, embeddings, strict=False):
            fitted_embedding, adjusted = self._fit_embedding_dimension(embedding, target_dimension)
            dimension_adjusted = dimension_adjusted or adjusted
            evidence_id = item.get("evidence_id", self._stable_id("E", item.get("summary", "")))
            metadata = {
                "case_id": case_id,
                "evidence_id": evidence_id,
                "doc_type": item.get("doc_type", "other"),
                "evidence_type": item.get("evidence_type", "other"),
                "party": item.get("party") or "",
                "summary": item.get("summary", ""),
                "raw_quote": item.get("raw_quote", ""),
                "issue_tags": item.get("issue_tags", []),
                "confidence": float(item.get("confidence", 0.5)),
            }
            vectors.append({"id": f"{case_id}_{evidence_id}", "values": fitted_embedding, "metadata": metadata})
        try:
            index.upsert(vectors=vectors, namespace="case-evidence")
            return {
                "status": "completed",
                "count": len(vectors),
                "namespace": "case-evidence",
                "source_dimension": len(embeddings[0]),
                "index_dimension": target_dimension,
                "dimension_adjusted": dimension_adjusted,
            }
        except Exception as exc:  # noqa: BLE001
            self.last_error = f"Pinecone evidence upsert failed: {exc}"
            return {"status": "failed", "count": 0, "reason": self.last_error}

    async def query_case_evidence(
        self,
        query: str,
        case_id: str,
        top_k: int = 5,
        filters: dict | None = None,
    ):
        index = self.index or self._connect_index()
        if index is None or not self.upstage.enabled:
            return []
        embedding = (await self.upstage.embed([query]))[0]
        target_dimension = self._index_dimension(index)
        embedding, _ = self._fit_embedding_dimension(embedding, target_dimension)
        pinecone_filter: dict[str, Any] = {"case_id": {"$eq": case_id}}
        if filters:
            pinecone_filter.update(filters)
        result = index.query(
            vector=embedding,
            top_k=top_k,
            namespace="case-evidence",
            filter=pinecone_filter,
            include_metadata=True,
        )
        return [match.get("metadata", {}) for match in result.get("matches", [])]

    async def upsert_legal_knowledge(self, legal_items: list):
        if not legal_items:
            return {"status": "skipped", "count": 0, "reason": "No legal knowledge items"}
        if not self.upstage.enabled:
            return {"status": "skipped", "count": 0, "reason": "UPSTAGE_API_KEY is not configured"}
        if not self.api_key:
            return {"status": "skipped", "count": 0, "reason": "PINECONE_API_KEY is not configured"}

        chunk_records = []
        for item in legal_items:
            full_text = "\n".join(
                [
                    item.get("title", ""),
                    item.get("summary", ""),
                    item.get("facts", ""),
                    item.get("holding", ""),
                    item.get("reasoning", ""),
                    item.get("content", ""),
                    item.get("full_text", ""),
                ]
            )
            chunks = self._chunk_text(full_text)
            for chunk_index, chunk_text in enumerate(chunks):
                chunk_records.append((item, chunk_index, len(chunks), chunk_text))

        texts = [record[3] for record in chunk_records]
        embeddings = await self.upstage.embed(texts)
        if not embeddings:
            return {"status": "failed", "count": 0, "reason": "Upstage embedding returned no vectors"}
        index = self._ensure_index(len(embeddings[0]))
        if index is None:
            return {"status": "failed", "count": 0, "reason": self.last_error}
        target_dimension = self._index_dimension(index)
        dimension_adjusted = bool(target_dimension and target_dimension != len(embeddings[0]))

        vectors = []
        for (item, chunk_index, chunk_count, chunk_text), embedding in zip(chunk_records, embeddings, strict=False):
            fitted_embedding, adjusted = self._fit_embedding_dimension(embedding, target_dimension)
            dimension_adjusted = dimension_adjusted or adjusted
            legal_id = item["legal_basis_id"]
            metadata = {
                "legal_basis_id": legal_id,
                "chunk_id": f"{legal_id}_CHUNK_{chunk_index + 1:03d}",
                "chunk_index": chunk_index,
                "chunk_count": chunk_count,
                "source_type": item.get("source_type", ""),
                "law_name": item.get("law_name") or "",
                "article_number": item.get("article_number") or "",
                "case_number": item.get("case_number") or "",
                "court": item.get("court") or "",
                "decision_date": item.get("decision_date") or "",
                "issue_type": item.get("issue_type", ""),
                "title": item.get("title", ""),
                "summary": item.get("summary", ""),
                "chunk_text": chunk_text,
                "source": item.get("source", ""),
                "source_url": item.get("source_url", ""),
            }
            vectors.append({"id": metadata["chunk_id"], "values": fitted_embedding, "metadata": metadata})
        try:
            index.upsert(vectors=vectors, namespace="legal-knowledge")
            return {
                "status": "completed",
                "count": len(vectors),
                "source_item_count": len(legal_items),
                "namespace": "legal-knowledge",
                "source_dimension": len(embeddings[0]),
                "index_dimension": target_dimension,
                "dimension_adjusted": dimension_adjusted,
            }
        except Exception as exc:  # noqa: BLE001
            self.last_error = f"Pinecone legal knowledge upsert failed: {exc}"
            return {"status": "failed", "count": 0, "reason": self.last_error}

    async def reset_legal_knowledge(self) -> dict[str, Any]:
        index = self.index or self._connect_index()
        if index is None:
            return {"status": "failed", "reason": self.last_error}
        try:
            index.delete(delete_all=True, namespace="legal-knowledge")
            return {"status": "completed", "namespace": "legal-knowledge"}
        except Exception as exc:  # noqa: BLE001
            self.last_error = f"Pinecone legal knowledge reset failed: {exc}"
            return {"status": "failed", "reason": self.last_error}

    async def query_legal_knowledge(self, query: str, issue_type: str | None = None, top_k: int = 5):
        index = self.index or self._connect_index()
        if index is None or not self.upstage.enabled:
            return []
        embedding = (await self.upstage.embed([query]))[0]
        target_dimension = self._index_dimension(index)
        embedding, _ = self._fit_embedding_dimension(embedding, target_dimension)
        pinecone_filter = {"issue_type": {"$eq": issue_type}} if issue_type else None
        result = index.query(
            vector=embedding,
            top_k=top_k,
            namespace="legal-knowledge",
            filter=pinecone_filter,
            include_metadata=True,
        )
        return [match.get("metadata", {}) for match in result.get("matches", [])]
