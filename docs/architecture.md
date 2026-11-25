# 폴더 구조

## 범위와 현재 상태

개인 이력 및 프로젝트 정보를 기반으로 답하는 챗봇을 만든다. LangChain으로 RAG 흐름을 구성하고 ChromaDB에 문서 임베딩을 저장한다.

ChromaDB Docker 실행 환경, Markdown 문서 적재와 유사도 검색, 검색 tool을 사용하는 LangChain 단일 에이전트를 구현했다. 임베딩은 OpenAI `text-embedding-3-small`, 답변 모델은 `gpt-5-mini`다. `test/test_ingest.py`는 적재·검색을, `test/test_agent.py`는 tool 사용과 답변을 검사한다. API 서버·웹 UI·세션 기억은 아직 구현하지 않았다.

## 디렉터리

```text
jonghwa_agent/
├── AGENTS.md
├── README.md
├── pyproject.toml
├── compose.yaml
├── .env.example
├── .gitignore
├── docs/
│   ├── architecture.md
│   ├── ingest-validation.md
│   └── agent-validation.md
├── test/
│   ├── test_ingest.py
│   ├── test_agent.py
│   └── reports/                 # 로컬 평가 보고서, Git 제외
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
        ├── tools.py
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
| `compose.yaml` | ChromaDB 컨테이너 실행, 로컬 포트와 데이터 저장 경로 설정 |
| `.env.example` | 필요한 환경변수의 예시. 실제 값은 `.env`에서 관리 |
| `docs/architecture.md` | 폴더별 책임과 처리 흐름 기록 |
| `docs/ingest-validation.md` | 청킹·적재 방식과 검색 평가 기준 |
| `test/test_ingest.py` | 로컬 테스트, 실제 DB 검사, 질문 10개 검색 평가 |
| `test/test_agent.py` | tool 단위 테스트, 실제 모델의 tool 선택·답변 검사 |
| `docs/agent-validation.md` | 에이전트 실행 구조와 테스트 기준 |
| `data/documents/` | 이력서, 포트폴리오, 프로젝트 설명 등 원본 자료 |
| `data/chroma/` | ChromaDB 영속 저장소 |
| `src/jonghwa_agent/config.py` | 모델 및 경로 등 공통 설정 |
| `src/jonghwa_agent/ingest.py` | 원본 문서를 읽고 청크로 나누어 임베딩을 저장 |
| `src/jonghwa_agent/retriever.py` | 질문과 관련된 문서 청크 검색 |
| `src/jonghwa_agent/tools.py` | 기존 retriever를 호출하고 청크 본문·출처를 반환하는 검색 tool |
| `src/jonghwa_agent/chatbot.py` | LangChain 에이전트 생성, 한 질문 실행과 답변 출력 |
| `src/jonghwa_agent/main.py` | 향후 대화 실행 진입점의 자리. 현재 미구현 |
| `src/jonghwa_agent/prompts/system.txt` | 사용자를 대신해 답하는 방식과 근거 기반 답변 원칙 |

## 처리 흐름

자료 등록과 질문 응답을 분리한다. 질문할 때마다 원본 전체를 다시 임베딩하지 않는다.

```text
[자료 등록]
data/documents/
    → ingest.py: YAML 파싱 → 제목별 청킹 → 긴 구간 추가 분할 → 임베딩 생성
    → HTTP ChromaDB: 저장 (컨테이너 볼륨 data/chroma/)

[질문 응답]
chatbot.py: 질문 입력 → gpt-5-mini 에이전트
    → 필요 시 tools.py → retriever.py → ChromaDB 검색
    → 검색 본문·출처를 tool 응답으로 전달
    → 에이전트가 시스템 프롬프트·질문·검색 근거로 답변 생성
```

`config.py`는 두 흐름에서 필요한 설정을 제공한다. 개인 이력과 프로젝트 사실은 `data/documents/`에 두고, 말투와 답변 지침은 `prompts/system.txt`에 둔다.

## ChromaDB 실행 구조

- ChromaDB는 `chromadb/chroma:1.5.9` 이미지로 실행한다. 이미지 버전은 고정한다.
- 호스트의 `data/chroma/`를 컨테이너의 `/data`에 연결한다.
- 호스트의 `127.0.0.1:${CHROMA_PORT:-8000}`을 컨테이너의 `8000` 포트로 연결한다. `.env`는 Docker Compose가 읽는다.
- Python 코드는 로컬에서 실행하고 ChromaDB에는 HTTP 클라이언트로 연결한다. `config.py`가 `.env`와 실행 환경을 읽고 `retriever.py`가 공통 클라이언트·임베딩·검색을 제공한다.
- 인증 없이 로컬 개발에 사용하는 구성이다. 원격 배포 시 접근 제어는 별도로 설계한다.
- 영속 저장 경로와 HTTP 연결 방식은 [Chroma 공식 Docker 가이드](https://github.com/chroma-core/chroma/blob/1.5.9/docs/mintlify/guides/deploy/docker.mdx)를 기준으로 한다.

## 저장 규칙

- 실제 `.env`, 원본 자료 및 ChromaDB 데이터는 Git에서 제외한다.
- `data/` 아래의 `.gitkeep`은 빈 폴더 구조를 Git에 보존하기 위한 파일이다.
- API 서버와 웹 UI는 필요해질 때 추가한다. 현재 구조는 로컬 대화를 기준으로 한다.
