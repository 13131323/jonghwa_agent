"""LangChain 에이전트가 필요에 따라 검색 tool을 호출하고 답변한다."""

import argparse
from importlib.resources import files

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from jonghwa_agent.config import Settings, get_settings
from jonghwa_agent.tools import create_personal_search_tool


def create_personal_agent(settings: Settings | None = None):
    settings = settings or get_settings()
    settings.require_api_key()
    model = ChatOpenAI(
        model=settings.chat_model,
        api_key=settings.api_key,
        reasoning_effort="low",
        max_completion_tokens=4000,
        timeout=60,
        max_retries=2,
    )
    prompt = files("jonghwa_agent").joinpath("prompts/system.txt").read_text(encoding="utf-8")
    return create_agent(model=model, tools=[create_personal_search_tool(settings)], system_prompt=prompt)


def invoke_agent(agent, question: str):
    """독립된 한 질문을 실행하고 tool 호출·결과·최종 답변 메시지를 반환한다."""
    if not question.strip():
        raise ValueError("질문을 입력하세요.")
    return agent.invoke({"messages": [{"role": "user", "content": question}]},
                        config={"recursion_limit": 16})


def answer_text(result: dict) -> str:
    final = result["messages"][-1]
    if final.type != "ai" or getattr(final, "tool_calls", None):
        raise RuntimeError("에이전트가 최종 답변을 생성하지 못했습니다.")
    if isinstance(final.content, str):
        text = final.content
    else:
        text = "\n".join(block["text"] for block in final.content
                         if isinstance(block, dict) and block.get("type") == "text")
    if not text.strip():
        raise RuntimeError("에이전트의 답변이 비어 있습니다.")
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", help="에이전트에게 물을 질문")
    args = parser.parse_args()
    print(answer_text(invoke_agent(create_personal_agent(), args.question)))


if __name__ == "__main__":
    main()
