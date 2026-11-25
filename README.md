# jonghwa-agent

개인 이력·프로젝트 자료를 기반으로 답하는 AI 에이전트.

## 설치

Python 3.11 이상과 Docker Desktop이 필요합니다. 프로젝트 루트의 PowerShell에서 실행합니다.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

`.env`에 `OPENAI_API_KEY`와 `GEMINI_API_KEY`를 입력합니다.

## 실행

ChromaDB 컨테이너가 실행 중이어야 에이전트가 문서를 검색할 수 있습니다. Docker Desktop을 켜고 먼저 실행합니다.

```powershell
docker compose up -d
```

원본 Markdown을 `data/documents/`에 넣고 최초 실행 또는 문서 변경 시 적재합니다.

```powershell
python -m jonghwa_agent.ingest
```

위 컨테이너를 켜둔 상태에서 대화를 시작합니다.

```powershell
jonghwa-agent
```

## 종료

CLI는 `/exit` 또는 `Ctrl+C`로 종료합니다. 컨테이너 종료:

```powershell
docker compose down
```

상세 구조와 검증 방법은 [docs](docs/architecture.md)를 참고하세요.
