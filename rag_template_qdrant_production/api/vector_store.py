# Qdrantのコレクション作成、投入、検索を扱うベクトルDBアダプタです。
import uuid
from typing import Any, Dict, Iterable, Optional

from api.config import project_path


QDRANT_POINT_NAMESPACE = uuid.UUID("6f9f6068-bd8f-4b7d-a90a-8f0fc44c5578")


def qdrant_point_id(child_id: str) -> str:
    return str(uuid.uuid5(QDRANT_POINT_NAMESPACE, child_id))


def collection_name_from_settings(settings: dict[str, Any], collection_name: Optional[str] = None) -> str:
    if collection_name:
        return collection_name
    retrieval_settings = settings.get("retrieval", {})
    vector_settings = settings.get("vector_db", {})
    qdrant_settings = vector_settings.get("qdrant", {})
    return (
        qdrant_settings.get("collection_name")
        or retrieval_settings.get("collection_name")
        or f'{retrieval_settings.get("collection_prefix", "rag_")}children'
    )


class DenseVectorStore:
    provider = "base"

    def query(
        self,
        query_embedding: list[float],
        *,
        corpus_ids: Optional[list[str]] = None,
        limit: int = 20,
    ) -> Dict[str, float]:
        raise NotImplementedError

    def recreate(self, embedding_size: int) -> None:
        raise NotImplementedError

    def upsert(self, records: Iterable[dict[str, Any]]) -> None:
        raise NotImplementedError


class ChromaVectorStore(DenseVectorStore):
    provider = "chroma"

    def __init__(self, settings: dict[str, Any], collection_name: Optional[str] = None):
        import chromadb

        retrieval_settings = settings.get("retrieval", {})
        vector_settings = settings.get("vector_db", {})
        chroma_path = project_path(vector_settings.get("chroma_path") or retrieval_settings.get("chroma_path", "indexes/chroma"))
        self.collection_name = collection_name_from_settings(settings, collection_name)
        self.client = chromadb.PersistentClient(path=str(chroma_path))
        self.collection = self.client.get_or_create_collection(self.collection_name)

    def query(
        self,
        query_embedding: list[float],
        *,
        corpus_ids: Optional[list[str]] = None,
        limit: int = 20,
    ) -> Dict[str, float]:
        where = {"corpus_id": {"$in": list(corpus_ids)}} if corpus_ids is not None else None
        res = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            where=where,
            include=["metadatas", "documents", "distances"],
        )
        ids = res.get("ids", [[]])[0]
        distances = res.get("distances", [[]])[0]
        return {str(cid): 1.0 / (1.0 + float(dist)) for cid, dist in zip(ids, distances)}

    def recreate(self, embedding_size: int) -> None:
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(self.collection_name)

    def upsert(self, records: Iterable[dict[str, Any]]) -> None:
        records = list(records)
        if not records:
            return
        self.collection.add(
            ids=[r["child_id"] for r in records],
            documents=[r["document"] for r in records],
            metadatas=[r["payload"] for r in records],
            embeddings=[r["embedding"] for r in records],
        )


class QdrantVectorStore(DenseVectorStore):
    provider = "qdrant"

    def __init__(self, settings: dict[str, Any], collection_name: Optional[str] = None):
        from qdrant_client import QdrantClient

        self.settings = settings
        vector_settings = settings.get("vector_db", {})
        qdrant_settings = vector_settings.get("qdrant", {})
        self.collection_name = collection_name_from_settings(settings, collection_name)
        self.distance = str(qdrant_settings.get("distance", "cosine")).lower()
        self.payload_indexes = list(qdrant_settings.get("payload_indexes", ["corpus_id", "parent_id", "source_file"]))
        timeout_sec = int(qdrant_settings.get("timeout_sec", 60))
        local_path = str(qdrant_settings.get("local_path", "") or "").strip()
        if local_path:
            self.client = QdrantClient(path=str(project_path(local_path)))
        else:
            self.client = QdrantClient(
                url=str(qdrant_settings.get("url", "http://127.0.0.1:6333")),
                api_key=qdrant_settings.get("api_key") or None,
                prefer_grpc=bool(qdrant_settings.get("prefer_grpc", False)),
                timeout=timeout_sec,
            )

    def _models(self):
        from qdrant_client import models

        return models

    def _distance(self):
        models = self._models()
        if self.distance == "dot":
            return models.Distance.DOT
        if self.distance == "euclid":
            return models.Distance.EUCLID
        return models.Distance.COSINE

    def _filter(self, corpus_ids: Optional[list[str]]):
        if corpus_ids is None:
            return None
        models = self._models()
        return models.Filter(
            must=[
                models.FieldCondition(
                    key="corpus_id",
                    match=models.MatchAny(any=list(corpus_ids)),
                )
            ]
        )

    def query(
        self,
        query_embedding: list[float],
        *,
        corpus_ids: Optional[list[str]] = None,
        limit: int = 20,
    ) -> Dict[str, float]:
        query_filter = self._filter(corpus_ids)
        if hasattr(self.client, "query_points"):
            result = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
            points = getattr(result, "points", result)
        else:
            points = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )

        scores: Dict[str, float] = {}
        for point in points:
            payload = getattr(point, "payload", {}) or {}
            child_id = payload.get("child_id")
            if child_id:
                scores[str(child_id)] = float(getattr(point, "score", 0.0) or 0.0)
        return scores

    def recreate(self, embedding_size: int) -> None:
        models = self._models()
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(size=embedding_size, distance=self._distance()),
        )
        for field in self.payload_indexes:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
            except Exception:
                pass

    def upsert(self, records: Iterable[dict[str, Any]]) -> None:
        models = self._models()
        points = [
            models.PointStruct(
                id=qdrant_point_id(r["child_id"]),
                vector=r["embedding"],
                payload=r["payload"],
            )
            for r in records
        ]
        if not points:
            return
        self.client.upsert(collection_name=self.collection_name, points=points, wait=True)


def create_vector_store(settings: dict[str, Any], collection_name: Optional[str] = None) -> DenseVectorStore:
    provider = str(settings.get("vector_db", {}).get("provider", "chroma")).lower()
    if provider == "qdrant":
        return QdrantVectorStore(settings, collection_name=collection_name)
    if provider == "chroma":
        return ChromaVectorStore(settings, collection_name=collection_name)
    raise ValueError(f"unsupported vector_db.provider: {provider}")
