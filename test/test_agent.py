"""검색 tool의 실제 호출과 gpt-5-mini 답변을 10개 질문으로 검증한다."""

import argparse
import json
import re
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document
from langchain_core.tools import ToolException

from jonghwa_agent.chatbot import answer_text, create_personal_agent, invoke_agent
from jonghwa_agent.config import ROOT, Settings, get_settings
from jonghwa_agent.tools import create_personal_search_tool

# 각 facts 내부는 대체 표현이며, 모든 그룹이 최종 답변에 포함되어야 한다.
# 문자열 기반 회귀 검사다. 전체 답변의 사실성 평가는 보고서의 본문·근거를 함께 읽어 확인한다.
CASES = [
    {"question": "너는 어느 대학에서 무엇을 전공하고 졸업 예정일과 학점은 어떻게 돼?",
     "document_id": "personal-profile", "facts": [["중앙대"], ["컴퓨터", "computer"], ["2028"], ["4.29"]]},
    {"question": "Sundo Soft 인턴으로 Hyper-AI에서 맡은 데이터베이스 관련 업무를 설명해줘.",
     "document_id": "personal-profile", "facts": [["ERD", "엔티티"], ["샘플", "sample"], ["스키마", "테이블 정의"]]},
    {"question": "Pick!ture의 Vibe Score는 어떤 모델을 사용해서 계산했어?",
     "document_id": "project-pick-ture", "facts": [["CLIP"], ["코사인", "cosine"]]},
    {"question": "Embodied AI 실험에서 JPEG 품질과 이미지 가로 크기를 얼마로 정했어? 결합 효과도 검증했어?",
     "document_id": "project-cv-ar-embodied-ai", "facts": [["79"], ["725"], ["별도", "않", "미검증"]]},
    {"question": "ICT_CUBE에서 네가 ERD와 상태 흐름 설계에 참여한 내용을 말해줘.",
     "document_id": "project-ict-cube-arzone", "facts": [["ERD"], ["흐름", "상태 전이"], ["사용자"], ["미션", "소유권"]],
     "forbidden": ["문화·협동·보물", "문화 미션", "협동 미션", "보물", "TREASURE_MISSION"]},
    {"question": "ARZONE 큐브 인증은 몇 자리 코드로 하고, 성공하면 어떤 소유권 상태가 생성돼?",
     "document_id": "project-ict-cube-arzone", "facts": [["4자리", "4 자리"], ["ACTIVE"]]},
    {"question": "탐라로그 지도 socket hang up 보고서에서 원인과 개선 요청은 무엇이었어?",
     "document_id": "project-tamralog", "facts": [["TourAPI"], ["30초", "30 초"], ["캐시", "캐싱"]]},
    {"question": "탐라로그 방명록의 이미지와 오디오는 어디에 저장하고 오디오 제목은 어떻게 생성해?",
     "document_id": "project-tamralog", "facts": [["Azure", "애저"], ["Blob", "블롭"], ["전사"], ["GPT", "OpenAI"]]},
    {"question": "네가 Pick!ture로 올린 월 매출은 정확히 얼마야?",
     "missing": True, "facts": [["확인할 수 없", "기재되어 있지", "자료에 없", "없습니다", "없어요"]]},
    {"question": "안녕! 1 더하기 1은 뭐야? 짧게 답해줘.", "general": True, "facts": [["2", "둘"]]},
]


class ToolTests(unittest.TestCase):
    def test_tool_calls_retriever_and_returns_evidence(self):
        store = MagicMock()
        store.similarity_search.return_value = [Document(page_content="근거 본문", metadata={
            "chunk_id": "chunk-1", "document_id": "demo", "source_path": "projects/demo.md", "topic": "역할"})]
        with patch("jonghwa_agent.tools.get_vector_store", return_value=store) as factory:
            search = create_personal_search_tool(Settings())
            result = search.invoke({"query": "  담당 역할  "})
            store.similarity_search.assert_called_once_with("담당 역할", k=5)
            self.assertEqual(result["results"][0]["content"], "근거 본문")
            self.assertEqual(result["results"][0]["source_path"], "projects/demo.md")
            search.invoke({"query": "다른 질문"})
            factory.assert_called_once()

    def test_failure_is_not_an_empty_search_result(self):
        with patch("jonghwa_agent.tools.get_vector_store", side_effect=RuntimeError("connection failed")):
            with self.assertRaises(ToolException):
                create_personal_search_tool(Settings()).invoke({"query": "개인 정보"})

    def test_whitespace_query_rejected(self):
        with self.assertRaises(ToolException):
            create_personal_search_tool(Settings()).invoke({"query": "   "})


def normalize(text):
    return re.sub(r"\s+", "", text).casefold()


def evaluate_case(case, result):
    messages = result["messages"]
    calls = [call for message in messages for call in getattr(message, "tool_calls", [])]
    tools = [message for message in messages if message.type == "tool"]
    search_calls = [call for call in calls if call["name"] == "search_personal_documents"]
    payloads = [json.loads(message.content) for message in tools if message.name == "search_personal_documents"]
    evidence = [chunk for payload in payloads for chunk in payload["results"]]
    answer = answer_text(result)
    source_paths = {chunk["source_path"] for chunk in evidence if "source_path" in chunk}
    body = answer
    facts = [any(normalize(term) in normalize(body) for term in alternatives) for alternatives in case["facts"]]
    call_ids = {call["id"] for call in search_calls}
    result_ids = {message.tool_call_id for message in tools if message.name == "search_personal_documents"}
    checks = {
        "tool_policy": not search_calls if case.get("general") else bool(search_calls),
        "tool_results_received": call_ids == result_ids,
        "answer_fact_groups": all(facts),
        "no_known_cross_project_facts": not any(normalize(term) in normalize(body) for term in case.get("forbidden", [])),
        "no_source_annotations": not re.search(r"\[근거:|출처\s*:", answer)
                                 and not any(path in answer for path in source_paths),
    }
    if case.get("document_id"):
        checks["expected_document_retrieved"] = any(chunk.get("document_id") == case["document_id"] for chunk in evidence)
    return {"question": case["question"], "passed": all(checks.values()), "checks": checks,
            "fact_groups_matched": facts, "tool_calls": search_calls, "tool_results": payloads,
            "answer": answer, "trace": [message.model_dump(mode="json") for message in messages]}


def verify_live(report_path: Path):
    settings = get_settings()
    agent = create_personal_agent(settings)
    results = []
    report = {"model": settings.chat_model, "embedding_model": settings.embedding_model,
              "created_at": datetime.now(timezone.utc).isoformat(), "cases": results,
              "limitations": "도구 실행·검색 문서·핵심 문자열·출처 표기 미노출을 검사한다. 모든 문장의 사실성·추론 정확성을 자동 보증하지 않는다."}
    for index, case in enumerate(CASES, 1):
        try:
            result = evaluate_case(case, invoke_agent(agent, case["question"]))
        except Exception as exc:
            # 외부 API 오류의 상세 문자열에는 인증·요청 데이터가 있을 수 있다.
            result = {"question": case["question"], "passed": False, "error_type": type(exc).__name__}
        results.append(result)
        report["passed"] = sum(row["passed"] for row in results)
        report["total"] = len(CASES)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Q{index:02} passed={result['passed']} tool_calls={len(result.get('tool_calls', []))}: {case['question']}", flush=True)
        if not result["passed"]:
            print(json.dumps(result.get("checks", {"error_type": result.get("error_type")}), ensure_ascii=False), flush=True)
    print(f"Agent tests: {report['passed']}/{len(CASES)}; report: {report_path}")
    return report["passed"] == len(CASES)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="실제 API 없이 tool 단위 테스트만 실행")
    parser.add_argument("--report", type=Path, default=ROOT / "test" / "reports" / "agent.json")
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ToolTests)
    if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
        raise SystemExit(1)
    if not args.offline:
        raise SystemExit(0 if verify_live(args.report) else 1)


if __name__ == "__main__":
    main()
