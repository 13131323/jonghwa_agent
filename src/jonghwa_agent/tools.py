"""에이전트가 개인 자료를 조회하는 읽기 전용 검색 tool."""

from langchain_core.tools import ToolException, tool
from pydantic import BaseModel, Field

from jonghwa_agent.config import Settings, get_settings
from jonghwa_agent.retriever import get_vector_store


class PersonalSearchInput(BaseModel):
    query: str = Field(min_length=1, max_length=1000, description="찾을 개인 이력·프로젝트·경험에 대한 구체적인 검색어")


def create_personal_search_tool(settings: Settings | None = None):
    settings = settings or get_settings()
    store = None

    @tool("search_personal_documents", args_schema=PersonalSearchInput)
    def search_personal_documents(query: str) -> dict:
        """사용자의 학력, 경력, 역할, 프로젝트, 기술 선택, 실험 결과를 개인 문서에서 검색한다.

        일반 지식으로 알 수 없는 사용자 고유 사실을 답하기 전에 호출한다.
        구체적인 프로젝트명과 질문 주제를 query에 넣는다. 본문·제목·출처가 포함된
        상위 5개 청크를 반환한다. 검색 결과가 있어도 질문의 답이 없을 수 있으므로
        본문을 확인해야 한다. 근거가 부족하면 검색어를 바꾸어 다시 검색할 수 있다.
        """
        nonlocal store
        if not query.strip():
            raise ToolException("비어 있지 않은 구체적인 검색어가 필요합니다.")
        try:
            if store is None:
                store = get_vector_store(settings)
            docs = store.similarity_search(query.strip(), k=settings.top_k)
        except Exception as exc:
            # 연결 실패를 '자료 없음'으로 바꾸지 않는다. 원문 예외에 인증 정보가 있을 수 있다.
            raise ToolException("개인 문서 검색에 실패했습니다. ChromaDB 연결과 API 설정을 확인하세요.") from exc
        return {
            "query": query.strip(),
            "results": [
                {"content": doc.page_content, "chunk_id": doc.metadata.get("chunk_id", doc.id),
                 **{key: doc.metadata[key] for key in (
                     "document_id", "source_path", "source_url", "source_file", "title", "section", "topic"
                 ) if key in doc.metadata}}
                for doc in docs
            ],
        }

    return search_personal_documents
