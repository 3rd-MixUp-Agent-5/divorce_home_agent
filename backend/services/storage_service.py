import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any


class StorageService:
    def __init__(self, base_dir: str | None = None):
        self.base_dir = Path(base_dir or os.getenv("DATA_DIR", "./data"))
        self.cases_dir = self.base_dir / "cases"
        self.legal_dir = self.base_dir / "legal_knowledge"
        self.cases_dir.mkdir(parents=True, exist_ok=True)
        self.legal_dir.mkdir(parents=True, exist_ok=True)

    def _case_dir(self, case_id: str) -> Path:
        return self.cases_dir / case_id

    def create_case(self, case_info: dict) -> str:
        case_id = f"CASE_{uuid.uuid4().hex[:8].upper()}"
        case_dir = self._case_dir(case_id)
        (case_dir / "uploads").mkdir(parents=True, exist_ok=True)
        self.save_json(case_id, "case_info.json", case_info)
        self.save_json(case_id, "uploaded_files.json", [])
        return case_id

    def save_uploaded_file(self, case_id: str, file, doc_type: str | None = None) -> dict:
        case_dir = self._case_dir(case_id)
        upload_dir = case_dir / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)

        uploaded_files = self.load_uploaded_files(case_id)
        file_id = f"FILE_{len(uploaded_files) + 1:03d}"
        safe_name = Path(file.filename or f"{file_id}.bin").name
        target_path = upload_dir / f"{file_id}_{safe_name}"

        with target_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        item = {
            "file_id": file_id,
            "file_name": safe_name,
            "doc_type": doc_type or "other",
            "path": str(target_path),
        }
        uploaded_files.append(item)
        self.save_json(case_id, "uploaded_files.json", uploaded_files)
        return item

    def load_case_info(self, case_id: str) -> dict:
        return self.load_json(case_id, "case_info.json")

    def load_uploaded_files(self, case_id: str) -> list:
        try:
            return self.load_json(case_id, "uploaded_files.json")
        except FileNotFoundError:
            return []

    def save_json(self, case_id: str, filename: str, data: Any):
        case_dir = self._case_dir(case_id)
        case_dir.mkdir(parents=True, exist_ok=True)
        path = case_dir / filename
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_json(self, case_id: str, filename: str):
        path = self._case_dir(case_id) / filename
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def save_report(self, case_id: str, report: dict):
        self.save_json(case_id, "report.json", report)

    def load_report(self, case_id: str):
        return self.load_json(case_id, "report.json")

    def load_legal_basis(self) -> list:
        items = []
        seen = set()
        for path in sorted(self.legal_dir.glob("*.json")):
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                data = data.get("items", [])
            for item in data:
                legal_id = item.get("legal_basis_id")
                if legal_id and legal_id not in seen:
                    seen.add(legal_id)
                    items.append(item)
        if not items:
            raise FileNotFoundError("No legal knowledge JSON files found")
        return items
