"""로컬 .env와 실행 환경에서 공통 설정을 읽는다."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    documents_dir: Path = ROOT / "data" / "documents"
    chroma_host: str = "127.0.0.1"
    chroma_port: int = 8000
    collection_name: str = "jonghwa_documents"
    embedding_model: str = "text-embedding-3-small"
    api_key: str = field(default="", repr=False)
    chunk_size: int = 1000
    chunk_overlap: int = 150
    separators: tuple[str, ...] = ("\n\n", "\n", " ", "")
    top_k: int = 5

    def __post_init__(self):
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("0 <= chunk_overlap < chunk_size 이어야 합니다.")
        if self.top_k < 1:
            raise ValueError("top_k는 1 이상이어야 합니다.")

    def require_api_key(self):
        if not self.api_key:
            raise ValueError(".env의 OPENAI_API_KEY를 입력하세요.")


def get_settings() -> Settings:
    load_dotenv(ROOT / ".env", override=False)
    return Settings(
        chroma_host=os.getenv("CHROMA_HOST", "127.0.0.1"),
        chroma_port=int(os.getenv("CHROMA_PORT", "8000")),
        collection_name=os.getenv("CHROMA_COLLECTION", "jonghwa_documents"),
        embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        api_key=os.getenv("OPENAI_API_KEY", "").strip(),
    )
