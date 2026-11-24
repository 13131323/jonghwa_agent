# jonghwa_agent
개인 이력 및 프로젝트 정보를 기반으로 답변하는 AI 챗봇 에이전트

## ChromaDB 실행

Docker Desktop을 Linux 컨테이너 모드로 실행한 뒤 프로젝트 루트에서 실행합니다.

```powershell
docker compose up -d
```

기본 주소는 `http://127.0.0.1:8000`입니다. 포트를 바꾸려면 `.env.example`을 `.env`로 복사하고 `CHROMA_PORT`를 수정합니다. 기존 `.env`가 있으면 해당 항목만 추가합니다. `.env` 없이도 기본 포트로 실행됩니다.

상태와 연결 확인:

```powershell
docker compose ps
Invoke-RestMethod http://127.0.0.1:8000/api/v2/heartbeat
```

포트를 변경했다면 확인 주소에도 같은 포트를 사용합니다. 정상 응답에는 `nanosecond heartbeat` 값이 포함됩니다. 기동 직후 연결이 실패하면 잠시 후 다시 확인합니다.

로그 확인 및 종료:

```powershell
docker compose logs --tail 50 chroma
docker compose down
```

DB 데이터는 `data/chroma/`에 저장되므로 컨테이너를 종료·재생성해도 유지됩니다. 이 폴더를 삭제하면 저장된 데이터가 사라집니다. DB는 로컬에서만 접근할 수 있습니다.

현재는 DB 실행 환경만 준비되어 있으며, 문서 등록 및 챗봇 기능은 아직 구현하지 않았습니다. 폴더별 역할은 [아키텍처 문서](docs/architecture.md)를 참고하세요.
