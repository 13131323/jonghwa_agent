"""프로세스와 에이전트를 유지하는 jonghwa-agent 대화형 CLI."""

import argparse
import sys


def run_cli() -> int:
    print("jonghwa-agent를 시작합니다...", flush=True)
    try:
        from jonghwa_agent.chatbot import answer_text, create_personal_agent, invoke_agent

        agent = create_personal_agent()
    except KeyboardInterrupt:
        print("\n종료합니다.")
        return 0
    except Exception as exc:
        print(f"초기화 실패 ({type(exc).__name__}). .env와 설치 상태를 확인해 주세요.", file=sys.stderr)
        return 1

    print("질문을 입력하세요. /exit 또는 /quit으로 종료합니다.")
    print("현재 각 질문은 독립적으로 처리합니다.\n", flush=True)
    while True:
        try:
            question = input("나 > ").strip()
            if not question:
                continue
            if question.casefold() in {"/exit", "/quit"}:
                break
            print("답변 생성 중...", flush=True)
            try:
                result = invoke_agent(agent, question)
                print(f"\njonghwa-agent > {answer_text(result)}\n", flush=True)
            except Exception as exc:
                print(f"질문 처리 실패 ({type(exc).__name__}). 다시 입력해 주세요.\n", file=sys.stderr, flush=True)
        except (EOFError, KeyboardInterrupt):
            print()
            break
    print("종료합니다.")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    raise SystemExit(run_cli())


if __name__ == "__main__":
    main()
