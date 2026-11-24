# 폴더 구조

## 범위와 현재 상태

개인 이력 및 프로젝트 정보를 기반으로 답하는 챗봇을 만든다. LangChain으로 RAG 흐름을 구성하고 ChromaDB에 문서 임베딩을 저장한다.

현재는 폴더, 모듈 자리와 기본 프롬프트만 만들었다. 문서 등록, 검색, 모델 호출, 대화 실행은 아직 구현하지 않았다. LLM·임베딩 제공자, 의존성 버전, API 서버 및 웹 UI는 정하지 않았다.

## 디렉터리

```text
jonghwa_agent/
├── AGENTS.md
├── README.md
├── pyproject.toml
├── .env.example
├── .gitignore
├── docs/
│   └── architecture.md
├── data/
│   ├── documents/
│   │   └── .gitkeep
│   └── chroma/
│       └── .gitkeep
└── src/
    └── jonghwa_agent/
        ├── __init__.py
        ├── config.py
        ├── ingest.py
        ├── retriever.py
        ├── chatbot.py
        ├── main.py
        └── prompts/
            └── system.txt
```

## 역할

| 경로 | 역할 |
| --- | --- |
| `AGENTS.md` | 이 저장소에서 작업하는 개발 에이전트의 지침 |
| `README.md` | 구현 후 설치 및 실행 방법 안내 |
| `pyproject.toml` | Python 프로젝트 정보 및 의존성 정의 |
| `.env.example` | 필요한 환경변수의 예시. 실제 값은 `.env`에서 관리 |
| `docs/architecture.md` | 폴더별 책임과 처리 흐름 기록 |
| `data/documents/` | 이력서, 포트폴리오, 프로젝트 설명 등 원본 자료 |
| `data/chroma/` | ChromaDB 영속 저장소 |
| `src/jonghwa_agent/config.py` | 모델 및 경로 등 공통 설정 |
| `src/jonghwa_agent/ingest.py` | 원본 문서를 읽고 청크로 나누어 임베딩을 저장 |
| `src/jonghwa_agent/retriever.py` | 질문과 관련된 문서 청크 검색 |
| `src/jonghwa_agent/chatbot.py` | LangChain으로 검색 결과, 시스템 프롬프트, 질문을 연결하여 답변 생성 |
| `src/jonghwa_agent/main.py` | 로컬에서 질문을 입력하고 답변을 출력하는 진입점 |
| `src/jonghwa_agent/prompts/system.txt` | 사용자를 대신해 답하는 방식과 근거 기반 답변 원칙 |

## 구현할 처리 흐름

자료 등록과 질문 응답을 분리한다. 질문할 때마다 원본 전체를 다시 임베딩하지 않는다.

```text
[자료 등록]
data/documents/
    → ingest.py: 문서 로딩 → 청크 분할 → 임베딩 생성
    → data/chroma/: 저장

[질문 응답]
main.py: 질문 입력
    → chatbot.py
        → retriever.py → ChromaDB에서 관련 자료 검색
        → 검색 자료 + prompts/system.txt + 질문 → LLM 답변 생성
    → main.py: 답변 출력
```

`config.py`는 두 흐름에서 필요한 설정을 제공한다. 개인 이력과 프로젝트 사실은 `data/documents/`에 두고, 말투와 답변 지침은 `prompts/system.txt`에 둔다.

## 저장 규칙

- 실제 `.env`, 원본 자료 및 ChromaDB 데이터는 Git에서 제외한다.
- `data/` 아래의 `.gitkeep`은 빈 폴더 구조를 Git에 보존하기 위한 파일이다.
- API 서버와 웹 UI는 필요해질 때 추가한다. 현재 구조는 로컬 대화를 기준으로 한다.
