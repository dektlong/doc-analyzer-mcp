#!/usr/bin/env python3
"""
PostgreSQL + pgvector Vector Store

Provides a ChromaDB-compatible collection/client interface backed by
PostgreSQL with the pgvector extension, for use in Cloud Foundry deployments.

The interface mirrors the subset of the ChromaDB API used by rag_server.py:
    client.create_collection(name, metadata) -> collection
    client.delete_collection(name)
    collection.add(documents, metadatas, ids)
    collection.query(query_texts, n_results, include) -> {documents, metadatas, distances}
    collection.count() -> int
    collection.get(include) -> {ids, documents, metadatas}
"""

import json
import logging
import uuid
from typing import Any, Dict, List, Optional

import psycopg2
import psycopg2.extras
from openai import OpenAI

logger = logging.getLogger(__name__)

# Default embedding dimension; overridden by EMBEDDING_DIMENSION env var or
# auto-detected from the first embedding call.
_DEFAULT_EMBEDDING_DIMENSION = 1536


class PgVectorCollection:
    """
    A single named collection stored as a PostgreSQL table.

    Uses the OpenAI-compatible embedding endpoint to vectorise documents and
    queries, then performs similarity search via pgvector's <=> operator.
    """

    def __init__(
        self,
        conn: psycopg2.extensions.connection,
        table_name: str,
        embedding_client: OpenAI,
        embedding_model: str,
        embedding_dimension: int,
    ) -> None:
        self._conn = conn
        self._table = table_name
        self._embedding_client = embedding_client
        self._embedding_model = embedding_model
        self._embedding_dimension = embedding_dimension
        self._ensure_schema()

    # ------------------------------------------------------------------
    # Schema management
    # ------------------------------------------------------------------

    def _ensure_schema(self) -> None:
        """Create the table and index if they do not already exist."""
        with self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._table} (
                    id          TEXT        PRIMARY KEY,
                    content     TEXT        NOT NULL,
                    embedding   vector({self._embedding_dimension}),
                    metadata    JSONB,
                    created_at  TIMESTAMPTZ DEFAULT NOW()
                )
                """
            )
            # Drop legacy IVFFlat index if it exists (requires lists >= row count,
            # so it breaks on small tables).
            cur.execute(
                f"DROP INDEX IF EXISTS {self._table}_emb_idx"
            )
            # HNSW works correctly at any dataset size and has better recall.
            cur.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {self._table}_emb_hnsw_idx
                    ON {self._table}
                    USING hnsw (embedding vector_cosine_ops)
                """
            )
        self._conn.commit()
        logger.debug("Schema ensured for table %s", self._table)

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    def _embed(self, texts: List[str]) -> List[List[float]]:
        """Call the embedding API and return a list of float vectors."""
        response = self._embedding_client.embeddings.create(
            model=self._embedding_model,
            input=texts,
            encoding_format="float",
        )
        return [item.embedding for item in response.data]

    # ------------------------------------------------------------------
    # Public ChromaDB-compatible API
    # ------------------------------------------------------------------

    def add(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> None:
        """Embed and upsert documents into the collection."""
        if not documents:
            return

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in documents]
        if metadatas is None:
            metadatas = [{} for _ in documents]

        embeddings = self._embed(documents)

        with self._conn.cursor() as cur:
            for doc_id, content, embedding, metadata in zip(
                ids, documents, embeddings, metadatas
            ):
                cur.execute(
                    f"""
                    INSERT INTO {self._table} (id, content, embedding, metadata)
                    VALUES (%s, %s, %s::vector, %s)
                    ON CONFLICT (id) DO UPDATE
                        SET content   = EXCLUDED.content,
                            embedding = EXCLUDED.embedding,
                            metadata  = EXCLUDED.metadata
                    """,
                    (doc_id, content, str(embedding), json.dumps(metadata)),
                )
        self._conn.commit()
        logger.debug("Upserted %d documents into %s", len(documents), self._table)

    def query(
        self,
        query_texts: List[str],
        n_results: int = 5,
        include: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Find the *n_results* most similar documents for each query.

        Returns a dict matching ChromaDB's query response shape:
            {documents: [[...]], metadatas: [[...]], distances: [[...]]}
        """
        if not query_texts:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

        query_embeddings = self._embed(query_texts)

        all_documents: List[List[str]] = []
        all_metadatas: List[List[Dict]] = []
        all_distances: List[List[float]] = []

        for query_embedding in query_embeddings:
            with self._conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                cur.execute(
                    f"""
                    SELECT
                        content,
                        metadata,
                        1 - (embedding <=> %s::vector) AS similarity
                    FROM {self._table}
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (str(query_embedding), str(query_embedding), n_results),
                )
                rows = cur.fetchall()

            # distance = 1 - cosine_similarity  (0 = identical, 2 = opposite)
            all_documents.append([row["content"] for row in rows])
            all_metadatas.append([row["metadata"] or {} for row in rows])
            all_distances.append([1.0 - float(row["similarity"]) for row in rows])

        return {
            "documents": all_documents,
            "metadatas": all_metadatas,
            "distances": all_distances,
        }

    def count(self) -> int:
        """Return the number of documents in the collection."""
        with self._conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {self._table}")
            return cur.fetchone()[0]

    def get(self, include: Optional[List[str]] = None) -> Dict[str, Any]:
        """Return all documents (without embeddings)."""
        with self._conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(f"SELECT id, content, metadata FROM {self._table}")
            rows = cur.fetchall()

        return {
            "ids": [row["id"] for row in rows],
            "documents": [row["content"] for row in rows],
            "metadatas": [row["metadata"] or {} for row in rows],
        }

    def delete(self, ids: List[str]) -> None:
        """Delete specific documents by ID."""
        if not ids:
            return
        with self._conn.cursor() as cur:
            cur.execute(
                f"DELETE FROM {self._table} WHERE id = ANY(%s)",
                (ids,),
            )
        self._conn.commit()
        logger.debug("Deleted %d document(s) from %s", len(ids), self._table)

    def clear(self) -> None:
        """Delete all rows from the collection (schema remains intact)."""
        with self._conn.cursor() as cur:
            cur.execute(f"DELETE FROM {self._table}")
        self._conn.commit()
        logger.info("Cleared all documents from %s", self._table)


class PgVectorClient:
    """
    A ChromaDB-compatible client backed by PostgreSQL + pgvector.

    Manages one or more PgVectorCollection instances, each mapped to a
    separate table in the same database.
    """

    def __init__(
        self,
        db_uri: str,
        embedding_client: OpenAI,
        embedding_model: str,
        embedding_dimension: int = _DEFAULT_EMBEDDING_DIMENSION,
    ) -> None:
        self._conn = psycopg2.connect(db_uri)
        self._embedding_client = embedding_client
        self._embedding_model = embedding_model
        self._embedding_dimension = embedding_dimension
        self._collections: Dict[str, PgVectorCollection] = {}
        logger.info(
            "PgVectorClient connected. Embedding model: %s (%d dims)",
            embedding_model,
            embedding_dimension,
        )

    def _table_name(self, collection_name: str) -> str:
        return collection_name.lower().replace("-", "_").replace(" ", "_")

    def create_collection(
        self,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PgVectorCollection:
        """
        Return a PgVectorCollection for *name*, creating the underlying table
        if it does not yet exist.  Unlike ChromaDB this does NOT fail if the
        collection already exists; it simply returns the existing one.
        """
        if name not in self._collections:
            self._collections[name] = PgVectorCollection(
                conn=self._conn,
                table_name=self._table_name(name),
                embedding_client=self._embedding_client,
                embedding_model=self._embedding_model,
                embedding_dimension=self._embedding_dimension,
            )
        return self._collections[name]

    def get_collection(self, name: str) -> PgVectorCollection:
        """Return an existing collection, creating it if necessary."""
        return self.create_collection(name)

    def delete_collection(self, name: str) -> None:
        """
        Clear all rows from the named collection.

        Intentionally does NOT drop the table so that the schema (including the
        vector index) is preserved for fast re-ingestion.
        """
        if name in self._collections:
            self._collections[name].clear()
        else:
            # Collection not yet loaded in memory; create it just to clear it
            self.create_collection(name).clear()
