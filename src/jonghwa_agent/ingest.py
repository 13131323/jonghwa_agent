"""Markdown → 제목별 청킹 → OpenAI 임베딩 → HTTP ChromaDB 적재."""

import argparse
import hashlib
import json
import math
from datetime import date, datetime
from pathlib import Path

import yaml
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from jonghwa_agent.config import Settings, get_settings
from jonghwa_agent.retriever import get_client, get_collection, get_embeddings

HEADERS = [("#", "title"), ("##", "section"), ("###", "topic")]
CHUNKER_VERSION = "markdown-headers-v1"


def parse_markdown(text: str):
    lines = text.lstrip("\ufeff").splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        raise ValueError("YAML frontmatter가 필요합니다.")
    end = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), None)
    if end is None:
        raise ValueError("YAML frontmatter 종료 구분자가 없습니다.")
    metadata = yaml.safe_load("".join(lines[1:end]))
    if not isinstance(metadata, dict) or not isinstance(metadata.get("document_id"), str):
        raise ValueError("frontmatter에 문자열 document_id가 필요합니다.")
    if not metadata["document_id"].strip():
        raise ValueError("document_id가 비어 있습니다.")
    return metadata, "".join(lines[end + 1:]).strip()


def normalize_metadata(metadata: dict) -> dict:
    result = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if not isinstance(key, str):
            raise ValueError("메타데이터 키는 문자열이어야 합니다.")
        if isinstance(value, (datetime, date)):
            value = value.isoformat()
        elif isinstance(value, (list, dict)):
            value = json.dumps(value, ensure_ascii=False, sort_keys=True)
        elif not isinstance(value, (str, bool, int, float)):
            raise ValueError(f"지원하지 않는 메타데이터 타입: {key}")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"유한하지 않은 메타데이터: {key}")
        result[key] = value
    return result


def load_chunks(settings: Settings) -> list[Document]:
    header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS, strip_headers=False)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=list(settings.separators),
        length_function=len,
        is_separator_regex=False,
    )
    chunks, seen = [], set()
    for path in sorted(settings.documents_dir.rglob("*.md")):
        raw, body = parse_markdown(path.read_text(encoding="utf-8-sig"))
        document_id = raw["document_id"]
        if document_id in seen:
            raise ValueError(f"중복 document_id: {document_id}")
        seen.add(document_id)
        if not body:
            raise ValueError(f"본문이 비어 있습니다: {path.name}")
        common = normalize_metadata(raw)
        common.update(
            source_path=path.relative_to(settings.documents_dir).as_posix(),
            embedding_model=settings.embedding_model,
            chunker_version=CHUNKER_VERSION,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        for section_index, section in enumerate(header_splitter.split_text(body)):
            # 각 제목 구간을 별도로 나누므로 서로 다른 구간 사이에는 overlap이 없다.
            for chunk_index, chunk in enumerate(splitter.split_documents([section])):
                metadata = {**common, **chunk.metadata,
                            "section_index": section_index, "chunk_index": chunk_index}
                identity = json.dumps(
                    [CHUNKER_VERSION, document_id, settings.chunk_size, settings.chunk_overlap,
                     settings.separators, section_index, chunk_index, chunk.page_content],
                    ensure_ascii=False,
                )
                chunk_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
                metadata["chunk_id"] = chunk_id
                chunks.append(Document(id=chunk_id, page_content=chunk.page_content, metadata=metadata))
    if not chunks:
        raise ValueError("적재할 Markdown 문서가 없습니다.")
    return chunks


def ingest(settings: Settings | None = None, *, dry_run: bool = False):
    settings = settings or get_settings()
    chunks = load_chunks(settings)  # 전체 원문 검증을 마친 뒤 DB를 변경한다.
    summary = {"documents": len({c.metadata["document_id"] for c in chunks}), "chunks": len(chunks)}
    if dry_run:
        return {**summary, "dry_run": True, "max_characters": max(len(c.page_content) for c in chunks)}
    embeddings = get_embeddings(settings)
    collection = get_collection(get_client(settings), settings, create=True)
    embedded, unchanged, deleted = 0, 0, 0
    for document_id in sorted({c.metadata["document_id"] for c in chunks}):
        current = [c for c in chunks if c.metadata["document_id"] == document_id]
        previous = collection.get(where={"document_id": document_id}, include=["documents", "metadatas"])
        old = dict(zip(previous["ids"], zip(previous["documents"], previous["metadatas"])))
        pending = [c for c in current if c.id not in old or old[c.id][0] != c.page_content]
        for offset in range(0, len(pending), 64):
            batch = pending[offset:offset + 64]
            vectors = embeddings.embed_documents([c.page_content for c in batch])
            collection.upsert(ids=[c.id for c in batch], embeddings=vectors,
                              documents=[c.page_content for c in batch], metadatas=[c.metadata for c in batch])
            embedded += len(batch)
        for c in current:
            if c.id in old and old[c.id][0] == c.page_content:
                if old[c.id][1] != c.metadata:
                    collection.update(ids=[c.id], metadatas=[c.metadata])
                unchanged += 1
        stale = sorted(set(old) - {c.id for c in current})
        if stale:
            collection.delete(ids=stale)  # 새 청크 저장 성공 후 이 문서의 사라진 청크만 제거한다.
            deleted += len(stale)
    return {**summary, "embedded": embedded, "unchanged": unchanged, "deleted_stale": deleted}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="API·DB 호출 없이 청킹만 확인")
    args = parser.parse_args()
    print(json.dumps(ingest(dry_run=args.dry_run), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
