"""Gemini 입력 판별과 최종 답변의 전화번호·동호수 마스킹."""

import json
import re
from importlib.resources import files

from google import genai
from google.genai import types
from langchain.agents.middleware import after_agent, before_agent
from langchain_core.messages import AIMessage
from pydantic import BaseModel, ConfigDict, StrictBool

from jonghwa_agent.config import Settings, get_settings

BLOCK_MESSAGE = "욕설이나 비방이 포함된 메시지는 에이전트에게 전달되지 않습니다."
ERROR_MESSAGE = "입력 검사 서비스를 이용할 수 없습니다. 잠시 후 다시 시도해 주세요."
PRIVACY_NOTICE = "개인정보는 가려질 수 있습니다."

# 하이픈·공백·점 구분 또는 붙여 쓴 국내 휴대전화 번호.
PHONE_PATTERN = re.compile(r"(?<!\d)(01[016789])[-.\s]?(\d{3,4})[-.\s]?(\d{4})(?!\d)")
# 도로명·건물 번호는 건드리지 않고 숫자로 된 건물 동·호수만 처리한다.
UNIT_PAIR_PATTERN = re.compile(r"(?<![가-힣\d])\d{1,4}\s*동\s*[,/·-]?\s*\d{1,5}\s*호")
BUILDING_PATTERN = re.compile(r"(?<![가-힣\d])\d{1,4}\s*동")
ROOM_PATTERN = re.compile(r"(?<![가-힣\d])\d{1,5}\s*호")


def mask_private_text(text: str) -> tuple[str, bool]:
    text, phones = PHONE_PATTERN.subn(lambda match: f"{match[1]}-xxxx-xx{match[3][-2:]}", text)
    text, pairs = UNIT_PAIR_PATTERN.subn("xxx동 xxx호", text)
    text, buildings = BUILDING_PATTERN.subn("xxx동", text)
    text, rooms = ROOM_PATTERN.subn("xxx호", text)
    return text, bool(phones + pairs + buildings + rooms)


@after_agent(name="mask_private_output")
def mask_private_output(state, runtime):
    """최종 AIMessage를 같은 ID로 교체하여 원문 답변이 중복으로 남지 않게 한다."""
    message = state["messages"][-1]
    if message.type != "ai" or getattr(message, "tool_calls", None):
        return None
    changed = False
    if isinstance(message.content, str):
        content, changed = mask_private_text(message.content)
        if changed and PRIVACY_NOTICE not in content:
            content += f"\n\n{PRIVACY_NOTICE}"
    else:
        content = []
        for block in message.content:
            if isinstance(block, str):
                masked, found = mask_private_text(block)
                content.append(masked)
                changed |= found
            elif isinstance(block, dict) and block.get("type") == "text":
                masked, found = mask_private_text(block["text"])
                content.append({**block, "text": masked})
                changed |= found
            else:
                content.append(block)
        texts = [block if isinstance(block, str) else block.get("text", "")
                 for block in content if isinstance(block, (str, dict))]
        if changed and not any(PRIVACY_NOTICE in text for text in texts):
            content.append({"type": "text", "text": PRIVACY_NOTICE})
    if not changed:
        return None
    return {"messages": [message.model_copy(update={"content": content})]}


class InputVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")
    profanity: StrictBool
    harassment: StrictBool

    @property
    def blocked(self):
        return self.profanity or self.harassment


class GeminiInputJudge:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.prompt = files("jonghwa_agent").joinpath("prompts/guardrail.txt").read_text(encoding="utf-8")

    def classify(self, text: str) -> InputVerdict:
        if not self.settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY가 필요합니다.")
        try:
            with genai.Client(api_key=self.settings.gemini_api_key,
                              http_options=types.HttpOptions(timeout=15000)) as client:
                response = client.models.generate_content(
                    model=self.settings.guardrail_model,
                    contents=json.dumps({"input": text}, ensure_ascii=False),
                    config=types.GenerateContentConfig(
                        system_instruction=self.prompt,
                        temperature=0,
                        max_output_tokens=256,
                        thinking_config=types.ThinkingConfig(thinking_level="minimal"),
                        response_mime_type="application/json",
                        response_json_schema=InputVerdict.model_json_schema(),
                    ),
                )
                # 빈 응답·유형 불일치를 허용 판정으로 처리하지 않는다.
                return InputVerdict.model_validate_json(response.text or "")
        except Exception:
            # 외부 예외 문자열에는 인증 정보가 포함될 수 있으므로 전달하지 않는다.
            raise RuntimeError("Gemini 입력 판별에 실패했습니다.") from None


def create_input_guardrail(settings: Settings | None = None):
    judge = GeminiInputJudge(settings or get_settings())

    @before_agent(can_jump_to=["end"], name="gemini_input_guardrail")
    def check_input(state, runtime):
        latest = next((message for message in reversed(state["messages"]) if message.type == "human"), None)
        if latest is None:
            return {"messages": [AIMessage(content=ERROR_MESSAGE)], "jump_to": "end"}
        if isinstance(latest.content, str):
            text = latest.content
        else:
            text = "\n".join(block["text"] for block in latest.content
                             if isinstance(block, dict) and block.get("type") == "text")
        try:
            verdict = judge.classify(text)
        except RuntimeError:
            return {"messages": [AIMessage(content=ERROR_MESSAGE)], "jump_to": "end"}
        if verdict.blocked:
            return {"messages": [AIMessage(content=BLOCK_MESSAGE)], "jump_to": "end"}
        return None

    return check_input
