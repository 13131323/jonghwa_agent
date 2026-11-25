"""OpenAI 임베딩과 HTTP ChromaDB를 사용하는 청크 검색."""

import chromadb
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from jonghwa_agent.config import Settings, get_settings


def get_client(settings: Settings):
    client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
    client.heartbeat()
    return client


def get_embeddings(settings: Settings):
    settings.require_api_key()
    return OpenAIEmbeddings(model=settings.embedding_model, api_key=settings.api_key)


def get_collection(client, settings: Settings, *, create: bool = False):
    if create:
        collection = client.get_or_create_collection(
            name=settings.collection_name,
            embedding_function=None,
            metadata={"embedding_model": settings.embedding_model},
            configuration={"hnsw": {"space": "cosine"}},
        )
    else:
        collection = client.get_collection(settings.collection_name, embedding_function=None)
    if (collection.metadata or {}).get("embedding_model") != settings.embedding_model:
        raise ValueError("컬렉션의 임베딩 모델이 다릅니다. 새 CHROMA_COLLECTION을 지정하세요.")
    return collection


def get_vector_store(settings: Settings | None = None):
    settings = settings or get_settings()
    client = get_client(settings)
    get_collection(client, settings)
    return Chroma(
        client=client,
        collection_name=settings.collection_name,
        embedding_function=get_embeddings(settings),
        create_collection_if_not_exists=False,
    )


def similarity_search(query: str, k: int | None = None):
    settings = get_settings()
    k = settings.top_k if k is None else k
    if not query.strip() or k < 1:
        raise ValueError("검색어와 1 이상의 k가 필요합니다.")
    return get_vector_store(settings).similarity_search(query, k=k)
