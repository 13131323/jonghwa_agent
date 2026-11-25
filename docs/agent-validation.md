# 검색 tool과 에이전트 검증

## 실행 구조

`chatbot.py`는 LangChain `create_agent`와 ChatOpenAI로 단일 에이전트를 만든다. 모델은 `.env`의 `OPENAI_CHAT_MODEL`이며 기본값은 `gpt-5-mini`다. 임베딩 모델은 기존 `text-embedding-3-small`을 유지한다.

질문 → 에이전트의 tool 선택 → `tools.py`의 `search_personal_documents(query)` → 기존 `retriever.py`의 Chroma 검색 객체 → 상위 5개 청크 본문·문서 ID·제목·출처 경로 → 에이전트 최종 답변의 흐름이다. tool은 DB를 읽기만 하며 적재를 실행하지 않는다.

개인 이력·경험·프로젝트 수치는 검색 후 답하도록 시스템 프롬프트에 명시한다. 인사나 일반 개념 설명에는 tool 호출을 요구하지 않는다. `tool_choice=required`나 키워드 라우터로 강제하지 않고 모델이 tool을 선택한다. 검색 실패는 근거 없음으로 변환하지 않고 오류로 처리한다.

답변에는 실제 사용한 출처 경로를 `[근거:projects/example.md]` 형식으로 표시한다. 팀 기능·개인 기여, 계획·완료를 구분하고, 검색 결과에 없는 정보는 확인할 수 없다고 답하도록 지시한다. 프롬프트 지시만으로 모든 질문의 도구 사용과 사실성을 보장하지는 않는다.

현재는 한 번의 독립 질문 실행을 제공한다. `invoke_agent`는 매번 새 메시지 목록으로 호출하며 세션 기억·다중 사용자·UI는 구현하지 않는다. 반환된 전체 메시지에는 모델의 tool 요청·실행 결과·최종 답변이 포함된다.

## 테스트

`test/test_agent.py`는 로컬 tool 단위 테스트 3개와 실제 모델·DB를 사용하는 질문 10개를 실행한다. `--offline`은 단위 테스트만 실행한다.

| 질문 분류 | 개수 | 검사 |
| --- | --- | --- |
| 개인 사실 | 8 | 실제 tool 호출과 응답 메시지, 예상 문서 검색, 핵심 사실 문자열, 예상 문서 인용 |
| 문서에 없는 월 매출 | 1 | 검색을 실행하고 확인할 수 없다고 답하는지 |
| 일반 인사·산수 | 1 | 검색 없이 답하는지 |

학력·인턴 업무·Pick!ture의 CLIP 점수·Embodied AI 입력 설정·ICT_CUBE 역할·ARZONE 인증·탐라로그 지도 오류와 오디오 저장을 질문한다. 모델에는 질문과 시스템 프롬프트, tool 정의만 전달하고 예상 문서 ID·정답 표현은 전달하지 않는다. 검색어도 모델이 작성하며 DB 전체에서 검색한다.

개인 질문은 각 핵심 사실 그룹의 대체 표현 중 하나 이상이 답변에 있어야 한다. 모델의 tool call ID와 ToolMessage ID를 비교하고 인용 경로가 실제 반환된 출처에 있는지도 검사한다. ICT_CUBE의 개인 역할 답변에 탐라로그의 문화·협동·보물 미션을 섞는 알려진 오류도 금지 표현으로 검사한다. 모든 질문을 통과해야 성공 종료한다. 중간 API 오류는 해당 질문을 실패 처리하며 보고서에 오류 타입을 기록한다.

`test/reports/agent.json`에는 질문별 검사 결과, tool 검색어·결과 청크, 최종 답변과 메시지 전체를 저장한다. 개인 자료가 포함되어 Git에서 제외된다. 이 테스트는 도구 실행·핵심 사실·출처의 회귀 검사이며, 단어 포함 여부만으로 답변의 모든 주장과 추론이 정확하다고 판정하지 않는다. 실제 답변과 도구 본문을 함께 검토해야 한다.

## 참고

- [LangChain create_agent](https://reference.langchain.com/python/langchain/agents/factory/create_agent)
- [LangChain tools](https://docs.langchain.com/oss/python/langchain/tools)
- [OpenAI GPT-5 mini](https://developers.openai.com/api/docs/models/gpt-5-mini)
