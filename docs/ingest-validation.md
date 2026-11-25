# 문서 적재와 검색 검증

## 청킹과 저장

- YAML frontmatter는 PyYAML로 별도 파싱한다. `document_id`는 필수이며 파일 간 중복을 금지한다.
- MarkdownHeaderTextSplitter는 `# → title`, `## → section`, `### → topic`으로 제목 메타데이터를 만든다. `project_name`은 원문의 값을 유지한다.
- 제목 구간별로 RecursiveCharacterTextSplitter를 적용한다. `chunk_size=1000`, `chunk_overlap=150`, `length_function=len`으로 문자 수를 측정한다. 토큰 수가 아니다.
- separators는 문단·줄바꿈·공백·문자 순서인 `["\n\n", "\n", " ", ""]`다. 긴 제목 구간 안에서만 목표 150자의 겹침이 적용되며 실제 겹침량은 경계에 따라 다르다.
- YAML 날짜는 ISO 문자열, 리스트·객체는 JSON 문자열로 바꾸고 null은 제외한다. 공통 메타데이터와 제목·청크 식별 정보를 병합한다.
- 문서 ID, 청킹 버전·설정, 구간·청크 순서와 본문으로 SHA-256 청크 ID를 만든다. 같은 원문은 같은 ID를 갖는다.
- OpenAI `text-embedding-3-small`의 기본 1536차원 임베딩을 사용한다. ChromaDB HTTP 컬렉션의 거리 기준은 cosine이다.
- 문서별 기존 ID와 본문을 비교하여 새 청크만 임베딩·upsert한다. 본문이 그대로면 메타데이터만 갱신한다. 새 청크 저장 후 해당 문서의 사라진 청크를 삭제한다. 전체 작업이 원자적 트랜잭션은 아니며 중간 실패 시 재실행으로 수렴한다.
- 파일 전체 삭제를 이유로 DB 문서를 자동 삭제하지 않는다. 검증은 원본과 DB의 초과·누락 청크를 보고한다.
- 컬렉션에 기록된 모델과 설정 모델이 다르면 중단한다. 다른 임베딩 모델을 실험할 때는 새 컬렉션 이름을 사용한다.

## 단일 검증 파일

`test/test_ingest.py`는 기본 실행 시 아래 검사를 모두 수행한다. `--offline`은 첫 번째 단계만 수행한다.

1. 로컬 테스트: frontmatter 정규화, 긴 구간 추가 분할·겹침, 제목 간 분리, 안정적인 ID, 중복 없는 재적재, 실패 시 기존 청크 보존과 수정 시 이전 청크 제거. DB 작업은 이 단계에서만 가짜 컬렉션·임베딩을 사용한다.
2. 실제 DB 검사: 모든 원본 청크와 DB의 ID·본문·메타데이터 일치, 누락·초과 청크, 1536차원·유한·비영 벡터를 확인한다. 문서별 한 청크를 실제 OpenAI로 재임베딩하여 저장 벡터와 cosine 유사도 0.99 이상인지 확인한다. 초기 실행에서 동일 본문도 단일 입력과 배치 입력 간 약 0.9983의 유사도를 관측하여 수치 변동을 허용한다. 본문·메타데이터는 허용 오차 없이 일치해야 한다.
3. 검색 평가: 프로필·Pick!ture·Embodied AI·ICT_CUBE·탐라로그에서 각각 2개씩, 총 10개 질문을 검색한다. 실제 retriever와 같은 모델·컬렉션·top-k=5를 쓰고 distance가 포함된 검색 결과를 저장한다. 검색 시 문서 필터·재랭킹·MMR은 적용하지 않는다.

질문별 예상 근거는 파일 안에 문서 ID와 제목 구간으로 지정했다. 복합 질문은 필요한 제목 구간을 여러 그룹으로 둔다. 평가 전 해당 제목이 원문에 실제 있는지도 검사한다.

| 지표 | 의미 |
| --- | --- |
| Hit rate@5 | 필요한 근거 중 하나 이상을 상위 5개에서 찾은 질문 비율 |
| All evidence rate@5 | 필요한 모든 근거 그룹을 찾은 질문 비율 |
| Evidence coverage@5 | 질문별 필요한 근거 그룹 중 검색된 그룹의 비율을 평균 |
| Evidence precision@5 | 검색 청크 중 지정 근거 구간에 해당하는 청크 비율을 평균 |
| MRR@5 | 첫 근거 청크 순위의 역수 평균. 못 찾으면 0 |
| Near duplicate pair ratio | 결과 청크 쌍 중 출처 행·공백을 제외한 본문 5-gram Jaccard 유사도 0.8 이상인 쌍의 비율 |
| Unique sections | 질문별 검색 결과가 속한 서로 다른 문서·제목 구간 수 |

초기 회귀 감지 기준은 전체 근거 확보 질문 비율 80% 이상, 평균 유사 청크 쌍 비율 20% 이하다. 무결성 오류나 검색 기준 미달 시 종료 코드 1을 반환한다. 이 기준과 정답 구간은 실험 설정이며 보편적인 품질 보증 기준이 아니다.

동일 프로젝트의 청크가 여러 개 검색되는 것 자체는 중복으로 판정하지 않는다. 제목 구간 일치와 문자열 유사도는 근거 검색의 대리 지표이므로 실제 본문이 질문에 충분한지는 보고서에서 확인한다. 의미만 비슷한 중복이나 생성 답변의 사실성은 이 검사로 확정하지 않는다. LLM 답변 생성 평가는 포함하지 않는다.

보고서에는 질문·순위·거리·문서 ID·제목·출처 경로·전체 청크 본문을 남겨 근거 누락과 유사 결과 반복을 직접 검토할 수 있도록 한다. 개인 자료가 포함되므로 `test/reports/`는 Git에서 제외한다.

## 참고

- [LangChain Chroma 연동](https://docs.langchain.com/oss/python/integrations/vectorstores/chroma)
- [LangChain OpenAI 임베딩](https://docs.langchain.com/oss/python/integrations/text_embedding/openai)
- [LangChain RecursiveCharacterTextSplitter](https://reference.langchain.com/python/langchain-text-splitters/character/RecursiveCharacterTextSplitter)
