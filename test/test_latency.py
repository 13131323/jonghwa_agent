"""질문 하나의 실제 에이전트 구간별 지연을 출력한다.

실행: python test/test_latency.py
질문 지정: python test/test_latency.py "내 전화번호를 알려줘"
실제 Gemini·OpenAI API를 호출하며 운영 코드는 수정하지 않는다.
"""

import argparse
from collections import defaultdict
from contextlib import ExitStack
from functools import wraps
from time import perf_counter
from unittest.mock import patch


class Timings:
    def __init__(self):
        self.rows = []
        self.counts = defaultdict(int)

    def wrap(self, original, label, *, model_details=False):
        @wraps(original)
        def measured(*args, **kwargs):
            self.counts[label] += 1
            number = self.counts[label]
            print(f"[시작] {label} #{number}", flush=True)
            started = perf_counter()
            status = "성공"
            detail = ""
            try:
                result = original(*args, **kwargs)
                elapsed = perf_counter() - started
                if model_details:
                    generations = getattr(result, "generations", [])
                    message = getattr(generations[0], "message", None) if generations else None
                    if message is not None:
                        detail = "tool 호출 결정" if message.tool_calls else "답변 생성"
                        usage = message.usage_metadata or {}
                        reasoning = usage.get("output_token_details", {}).get("reasoning")
                        detail += f"; 입력 토큰={usage.get('input_tokens', '?')}, 출력 토큰={usage.get('output_tokens', '?')}"
                        if reasoning is not None:
                            detail += f", reasoning 토큰={reasoning}"
                return result
            except Exception as exc:
                elapsed = perf_counter() - started
                status = f"실패({type(exc).__name__})"
                raise
            finally:
                self.rows.append({"label": label, "number": number, "seconds": elapsed,
                                  "status": status, "detail": detail})
                print(f"[완료] {label} #{number}: {elapsed:.3f}초 / {status}"
                      + (f" / {detail}" if detail else ""), flush=True)

        return measured

    def print_summary(self, imports, initialization, invocation):
        totals = defaultdict(float)
        for row in self.rows:
            totals[row["label"]] += row["seconds"]
        labels = ["Gemini 입력 판별", "GPT 호출", "검색 연결·초기화", "검색어 임베딩", "ChromaDB 조회", "출력 마스킹 훅"]
        print("\n구간별 합계")
        print(f"{'의존성 import':<22} {imports:8.3f}초")
        print(f"{'에이전트 생성':<22} {initialization:8.3f}초")
        for label in labels:
            print(f"{label:<22} {totals[label]:8.3f}초 ({self.counts[label]}회)")
        print(f"{'기타 에이전트 처리':<22} {max(0.0, invocation - sum(totals.values())):8.3f}초")
        print(f"{'질문 실행 전체':<22} {invocation:8.3f}초")
        print(f"{'import·생성 포함 전체':<22} {imports + initialization + invocation:8.3f}초")
        print("호출되지 않은 구간은 0회입니다. 네트워크 대기·SDK 재시도는 해당 호출 시간에 포함됩니다.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?", default="내가 인턴으로 어떤 일을 했어?")
    args = parser.parse_args()
    if not args.question.strip():
        parser.error("비어 있지 않은 질문이 필요합니다.")

    started = perf_counter()
    from chromadb.api.models.Collection import Collection
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from jonghwa_agent import tools
    from jonghwa_agent.chatbot import answer_text, create_personal_agent, invoke_agent
    from jonghwa_agent.guardrails import ERROR_MESSAGE, GeminiInputJudge, mask_private_output
    imports = perf_counter() - started

    timer = Timings()
    print(f"질문: {args.question}")
    print(f"의존성 import: {imports:.3f}초", flush=True)
    result, error = None, None
    initialization, invocation = 0.0, 0.0
    with ExitStack() as stack:
        # 프로덕션 함수는 그대로 실행하고, 이 프로세스 안에서만 시간을 계측한다.
        targets = [
            (GeminiInputJudge, "classify", "Gemini 입력 판별", False),
            (ChatOpenAI, "_generate", "GPT 호출", True),
            (tools, "get_vector_store", "검색 연결·초기화", False),
            (OpenAIEmbeddings, "embed_query", "검색어 임베딩", False),
            (Collection, "query", "ChromaDB 조회", False),
            (mask_private_output, "after_agent", "출력 마스킹 훅", False),
        ]
        for owner, attribute, label, details in targets:
            stack.enter_context(patch.object(owner, attribute,
                                              timer.wrap(getattr(owner, attribute), label, model_details=details)))
        started = perf_counter()
        try:
            agent = create_personal_agent()
        except Exception as exc:
            error = type(exc).__name__
        finally:
            initialization = perf_counter() - started
        if error is None:
            print(f"에이전트 생성: {initialization:.3f}초", flush=True)
            started = perf_counter()
            try:
                result = invoke_agent(agent, args.question)
            except Exception as exc:
                error = type(exc).__name__
            finally:
                invocation = perf_counter() - started

    timer.print_summary(imports, initialization, invocation)
    if error is not None:
        print(f"\n실행 오류: {error}")
        raise SystemExit(1)
    answer = answer_text(result)
    print(f"\n최종 답변:\n{answer}")
    if answer == ERROR_MESSAGE:
        print("\n입력 판별 서비스 오류로 조기 종료되었습니다. 정상 답변의 지연 측정 결과가 아닙니다.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
