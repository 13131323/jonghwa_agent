"""단일 검증 파일: --offline은 로컬 테스트, 기본 실행은 실제 DB·검색 평가."""

import argparse
import itertools
import json
import math
import re
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from jonghwa_agent.config import ROOT, Settings, get_settings
from jonghwa_agent.ingest import ingest, load_chunks, normalize_metadata, parse_markdown
from jonghwa_agent.retriever import get_client, get_collection, get_embeddings, get_vector_store

# 질문마다 '정답을 뒷받침할 제목 구간'을 수작업으로 지정한다.
# 한 그룹 내 제목은 대체 가능한 근거, 여러 그룹은 각각 필요한 근거다.
CASES = [
    {"question": "이종화는 어느 학교에서 무엇을 전공하고 졸업 예정일과 학점은 어떻게 돼?",
     "document_id": "personal-profile", "evidence": [["중앙대학교 컴퓨터공학 학사 과정"]]},
    {"question": "Sundo Soft 인턴으로 Hyper-AI에서 어떤 데이터베이스 설계 업무를 했어?",
     "document_id": "personal-profile", "evidence": [["Sundo Soft 소프트웨어 개발 인턴"]]},
    {"question": "Pick!ture에서 CLIP 기반 Vibe Score와 Aesthetic Score는 각각 어떻게 계산해?",
     "document_id": "project-pick-ture", "evidence": [["CLIP 임베딩과 Vibe Score"], ["Aesthetic Score"]]},
    {"question": "Pick!ture의 투표 API는 어떤 방식으로 투표를 저장해?",
     "document_id": "project-pick-ture", "evidence": [["투표 API와 저장 방식"]]},
    {"question": "Embodied AI 연구에서 이종화가 JPEG 품질과 이미지 크기 실험으로 선택한 설정과 한계는?",
     "document_id": "project-cv-ar-embodied-ai", "evidence": [["JPEG 품질과 이미지 크기 실험"]]},
    {"question": "Embodied AI에서 Metric Mask_norm과 raw floor_margin을 구분해서 사용한 이유는?",
     "document_id": "project-cv-ar-embodied-ai", "evidence": [["크기 특징과 상태 분류 특징의 역할 분리"]]},
    {"question": "ARZONE의 아티스트 대화 RAG는 문서를 어떻게 청킹하고 재색인 중복을 방지해?",
     "document_id": "project-ict-cube-arzone", "evidence": [["섹션 청킹과 멱등 재색인"]]},
    {"question": "ICT_CUBE에서 이종화가 ERD와 상태 흐름 설계에 참여한 내용은?",
     "document_id": "project-ict-cube-arzone", "evidence": [["ERD와 상태 흐름 설계"]]},
    {"question": "탐라로그 지도 API의 socket hang up 원인과 요청된 개선 방법은 무엇이야?",
     "document_id": "project-tamralog", "evidence": [["지도 요청 실패 원인 분석과 캐시 적용 요청"]]},
    {"question": "탐라로그 방명록의 이미지와 오디오는 어디에 저장하고 오디오 제목은 어떻게 생성해?",
     "document_id": "project-tamralog", "evidence": [["일반 방명록과 길바닥 방명록"], ["오디오 전사와 방명록 제목 생성"]]},
]


def heading(doc):
    return " / ".join(doc.metadata.get(key, "") for key in ("title", "section", "topic"))


def grams(text):
    text = re.sub(r"출처:.*", "", text)
    text = re.sub(r"\s+", "", text.lower())
    return {text[i:i + 5] for i in range(max(0, len(text) - 4))}


def near_duplicate_ratio(docs, threshold=0.8):
    """상위 결과의 본문 5-gram Jaccard 유사도가 0.8 이상인 쌍의 비율."""
    pairs = list(itertools.combinations([grams(d.page_content) for d in docs], 2))
    return sum(bool(a | b) and len(a & b) / len(a | b) >= threshold for a, b in pairs) / len(pairs) if pairs else 0.0


def cosine(a, b):
    return sum(float(x) * float(y) for x, y in zip(a, b)) / (
        math.sqrt(sum(float(x) ** 2 for x in a)) * math.sqrt(sum(float(y) ** 2 for y in b)))


class OfflineTests(unittest.TestCase):
    def test_frontmatter_and_metadata(self):
        meta, body = parse_markdown("---\ndocument_id: demo\nupdated_at: 2026-11-25\nsource_date: null\n---\n# 제목\n본문")
        clean = normalize_metadata(meta)
        self.assertEqual(clean["updated_at"], "2026-11-25")
        self.assertNotIn("source_date", clean)
        self.assertTrue(body.startswith("# 제목"))
        with self.assertRaises(ValueError):
            parse_markdown("# frontmatter 없는 문서")

    def test_chunk_boundaries_and_stable_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "demo.md"
            path.write_text("---\ndocument_id: demo\nvisibility: private\n---\n# 프로젝트\n## 영역\n### 첫 주제\n" + "가나다라 내용입니다. " * 200 + "\n### 둘째 주제\n서로다른본문", encoding="utf-8")
            settings = replace(Settings(), documents_dir=Path(directory))
            chunks = load_chunks(settings)
            self.assertEqual([d.id for d in chunks], [d.id for d in load_chunks(settings)])
            self.assertGreater(len(chunks), 2)
            self.assertTrue(all(len(d.page_content) <= 1000 for d in chunks))
            second = [d for d in chunks if d.metadata.get("topic") == "둘째 주제"]
            self.assertEqual(len(second), 1)
            self.assertNotIn("가나다라", second[0].page_content)
            self.assertEqual(second[0].metadata["visibility"], "private")
            # 같은 제목의 긴 구간에서는 실제로 겹치는 텍스트가 있는지도 확인한다.
            first = [d for d in chunks if d.metadata.get("topic") == "첫 주제"]
            self.assertTrue(any(a.page_content.endswith(b.page_content[:n])
                                for a, b in zip(first, first[1:]) for n in range(10, 151)))

    def test_reingest_and_failed_update(self):
        class MemoryCollection:
            def __init__(self):
                self.rows = {}
            def get(self, where, include):
                items = [(key, row) for key, row in self.rows.items() if row[1]["document_id"] == where["document_id"]]
                return {"ids": [key for key, _ in items], "documents": [row[0] for _, row in items], "metadatas": [row[1] for _, row in items]}
            def upsert(self, ids, documents, metadatas, embeddings):
                self.rows.update({key: (text, meta) for key, text, meta in zip(ids, documents, metadatas)})
            def update(self, ids, metadatas):
                for key, meta in zip(ids, metadatas):
                    self.rows[key] = (self.rows[key][0], meta)
            def delete(self, ids):
                for key in ids:
                    del self.rows[key]

        class FakeEmbeddings:
            def embed_documents(self, texts):
                return [[1.0, 0.0] for _ in texts]

        collection = MemoryCollection()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "demo.md"
            path.write_text("---\ndocument_id: demo\n---\n# 제목\n원래 내용", encoding="utf-8")
            settings = replace(Settings(), documents_dir=Path(directory))
            with patch("jonghwa_agent.ingest.get_client"), patch("jonghwa_agent.ingest.get_collection", return_value=collection), patch("jonghwa_agent.ingest.get_embeddings", return_value=FakeEmbeddings()):
                self.assertEqual(ingest(settings)["embedded"], 1)
                self.assertEqual(ingest(settings)["embedded"], 0)
                old_ids = set(collection.rows)
                path.write_text("---\ndocument_id: demo\n---\n# 제목\n수정된 새 내용", encoding="utf-8")
                with patch.object(FakeEmbeddings, "embed_documents", side_effect=RuntimeError("API failure")):
                    with self.assertRaises(RuntimeError):
                        ingest(settings)
                self.assertEqual(set(collection.rows), old_ids)
                self.assertEqual(ingest(settings)["deleted_stale"], 1)
                self.assertFalse(old_ids & set(collection.rows))

    def test_gold_cases_exist_in_source(self):
        chunks = load_chunks(get_settings())
        self.assertEqual(len(CASES), 10)
        for case in CASES:
            relevant = [d for d in chunks if d.metadata["document_id"] == case["document_id"]]
            for alternatives in case["evidence"]:
                self.assertTrue(any(any(label in heading(d) for label in alternatives) for d in relevant), case["question"])


def verify_live(report_path: Path):
    settings = get_settings()
    expected = {d.id: d for d in load_chunks(settings)}
    collection = get_collection(get_client(settings), settings)
    errors, actual = [], {}
    for offset in range(0, collection.count(), 100):
        batch = collection.get(limit=100, offset=offset, include=["documents", "metadatas", "embeddings"])
        for key, text, meta, vector in zip(batch["ids"], batch["documents"], batch["metadatas"], batch["embeddings"]):
            actual[key] = (text, meta, vector)
    if set(actual) != set(expected):
        errors.append(f"원문 대비 청크 불일치: missing={len(set(expected)-set(actual))}, extra={len(set(actual)-set(expected))}")
    for key, doc in expected.items():
        if key not in actual:
            continue
        text, meta, vector = actual[key]
        if text != doc.page_content or meta != doc.metadata:
            errors.append(f"본문/메타데이터 불일치: {key}")
        if len(vector) != 1536 or not all(math.isfinite(float(v)) for v in vector) or not any(vector):
            errors.append(f"1536차원 유한·비영 벡터 검증 실패: {key}")
    # 실제 OpenAI로 문서별 한 청크를 재임베딩해 저장 벡터와 비교한다.
    samples = list({d.metadata["document_id"]: d for d in expected.values() if d.id in actual}.values())
    embeddings = get_embeddings(settings)
    comparisons = []
    for doc in samples:
        vector = embeddings.embed_documents([doc.page_content])[0]
        similarity = cosine(vector, actual[doc.id][2])
        comparisons.append({"document_id": doc.metadata["document_id"], "cosine": similarity})
        # 실제 API에서 단일/배치 입력 간 약 0.9983의 유사도를 관측했다.
        # 외부 모델 수치 변동을 허용하되 다른 벡터 저장은 감지하는 허용 오차다.
        if similarity < 0.99:
            errors.append(f"재임베딩 벡터 불일치: {doc.metadata['document_id']}")
    store = get_vector_store(settings)
    results = []
    for case in CASES:
        # 검색 엔진·k·임베딩 모델은 실제 retriever와 동일하다. 점수는 cosine distance(작을수록 가까움).
        hits = store.similarity_search_with_score(case["question"], k=settings.top_k)
        docs = [d for d, _ in hits]
        group_ranks = []
        for alternatives in case["evidence"]:
            rank = next((i for i, doc in enumerate(docs, 1)
                         if doc.metadata["document_id"] == case["document_id"]
                         and any(label in heading(doc) for label in alternatives)), None)
            group_ranks.append(rank)
        first_rank = min((rank for rank in group_ranks if rank), default=None)
        coverage = sum(rank is not None for rank in group_ranks) / len(group_ranks)
        evidence_precision = sum(doc.metadata["document_id"] == case["document_id"] and
                                 any(label in heading(doc) for group in case["evidence"] for label in group)
                                 for doc in docs) / len(docs) if docs else 0.0
        result = {**case, "hit": first_rank is not None, "evidence_coverage": coverage,
                  "evidence_precision": evidence_precision, "reciprocal_rank": 1 / first_rank if first_rank else 0,
                  "all_evidence_found": coverage == 1, "near_duplicate_pair_ratio": near_duplicate_ratio(docs),
                  "unique_sections": len({(d.metadata["document_id"], heading(d)) for d in docs}),
                  "hits": [{"rank": i, "distance": float(score), "chunk_id": d.metadata["chunk_id"],
                            "document_id": d.metadata["document_id"], "heading": heading(d),
                            "source_path": d.metadata["source_path"], "content": d.page_content}
                           for i, (d, score) in enumerate(hits, 1)]}
        results.append(result)
        print(f"Q{len(results):02} hit={result['hit']} coverage={coverage:.0%} duplicate={result['near_duplicate_pair_ratio']:.0%}: {case['question']}")
    avg = lambda key: sum(r[key] for r in results) / len(results)
    metrics = {"hit_rate_at_5": avg("hit"), "all_evidence_rate_at_5": avg("all_evidence_found"),
               "mean_evidence_coverage_at_5": avg("evidence_coverage"), "mean_reciprocal_rank_at_5": avg("reciprocal_rank"),
               "mean_evidence_precision_at_5": avg("evidence_precision"), "mean_near_duplicate_pair_ratio": avg("near_duplicate_pair_ratio")}
    # 초기 회귀 감지 기준이며, 답변의 의미적 정확도를 보장하는 기준은 아니다.
    quality_passed = metrics["all_evidence_rate_at_5"] >= 0.8 and metrics["mean_near_duplicate_pair_ratio"] <= 0.2
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "collection": settings.collection_name,
              "embedding_model": settings.embedding_model, "chunk_size": settings.chunk_size,
              "chunk_overlap": settings.chunk_overlap, "k": settings.top_k, "expected_chunks": len(expected),
              "stored_chunks": len(actual), "integrity_errors": errors, "reembedding_samples": comparisons,
              "reembedding_cosine_min": 0.99,
              "quality_thresholds": {"all_evidence_rate_at_5_min": 0.8, "mean_near_duplicate_pair_ratio_max": 0.2},
              "quality_passed": quality_passed, "metrics": metrics, "queries": results,
              "limitation": "제목으로 정의한 근거 검색과 문자열 중복의 대리 평가이며 답변 생성·사실성 평가는 포함하지 않는다."}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"integrity_errors": errors, "metrics": metrics, "quality_passed": quality_passed}, ensure_ascii=False, indent=2))
    print(f"Report: {report_path}")
    return not errors and quality_passed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="API·ChromaDB 없이 로컬 테스트만 실행")
    parser.add_argument("--report", type=Path, default=ROOT / "test" / "reports" / "retrieval.json")
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(OfflineTests)
    if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
        raise SystemExit(1)
    if not args.offline:
        raise SystemExit(0 if verify_live(args.report) else 1)


if __name__ == "__main__":
    main()
