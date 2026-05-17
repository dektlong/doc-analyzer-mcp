#!/usr/bin/env python3
"""
RAG MCP Server - Retrieval Augmented Generation using FastMCP

Storage    : PostgreSQL + pgvector  (bound via VCAP_SERVICES)
Embeddings : OpenAI-compatible model (bound via VCAP_SERVICES)
Transport  : SSE/HTTP on $PORT (default 8080)

Documents are read from the documents/ directory at startup and on reingest.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, urlunparse

from fastmcp import FastMCP
from openai import OpenAI

from cf_config import get_model_config, get_postgres_uri, log_vcap_summary
from pg_vector_store import PgVectorClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

mcp = FastMCP("RAG Server")

DATA_DIR = Path("documents")

TEXT_EXTENSIONS = {
    ".txt", ".md", ".rst", ".csv", ".json",
    ".yaml", ".yml", ".toml", ".xml", ".html",
    ".htm", ".log", ".sql",
}


def _extract_text(path: Path) -> str:
    """Return the text content of a file, handling PDF separately."""
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        return "\n".join(
            page.extract_text() or "" for page in reader.pages
        ).strip()
    return path.read_text(encoding="utf-8", errors="replace").strip()


SUPPORTED_EXTENSIONS = TEXT_EXTENSIONS | {".pdf"}

CHUNK_SIZE = 1000      # characters per chunk
CHUNK_OVERLAP = 200    # overlap between consecutive chunks


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split *text* into overlapping fixed-size character chunks."""
    if not text:
        return []
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start += chunk_size - overlap
    return chunks


# Initialised in __main__ — None until the process actually starts
_server: Optional["RAGServer"] = None


# ---------------------------------------------------------------------------
# Backend — plain class, no MCP decorators
# ---------------------------------------------------------------------------

class RAGServer:
    def __init__(self) -> None:
        self._db_client: Optional[PgVectorClient] = None
        self.collection = None
        self._chat_model_config: Optional[Dict[str, Any]] = None
        self._initialize()

    def _initialize(self) -> None:
        # Always dump the VCAP_SERVICES structure first so we can debug
        log_vcap_summary()

        db_uri = get_postgres_uri()
        if not db_uri:
            raise RuntimeError(
                "No PostgreSQL binding found. "
                "Bind a PostgreSQL service to this app (cf bind-service ...)."
            )

        embedding_cfg = get_model_config("embedding")
        if not embedding_cfg or not embedding_cfg.get("api_base"):
            raise RuntimeError(
                "No embedding model binding found (api_base is missing). "
                "Check the VCAP_SERVICES dump above for the actual credential structure."
            )

        self._chat_model_config = get_model_config("chat")
        if not self._chat_model_config:
            logger.warning("No chat model binding found in VCAP_SERVICES.")

        embedding_client = OpenAI(
            base_url=embedding_cfg["api_base"],
            api_key=embedding_cfg["api_key"],
        )
        # Env vars let operators pick specific models when the plan provides many
        embedding_model = (
            os.getenv("EMBEDDING_MODEL_NAME")
            or embedding_cfg.get("model_name")
            or "text-embedding-ada-002"
        )
        embedding_dimension = int(os.getenv("EMBEDDING_DIMENSION", "1536"))

        self._db_client = PgVectorClient(
            db_uri=db_uri,
            embedding_client=embedding_client,
            embedding_model=embedding_model,
            embedding_dimension=embedding_dimension,
        )

        self.collection = self._db_client.create_collection(
            name="rag_documents",
            metadata={"description": "Collection for RAG document storage"},
        )

        logger.info(
            "Initialised. Collection has %d documents. Embedding model: %s",
            self.collection.count(),
            embedding_model,
        )
        self._ingest_data_dir()

    def _read_files(self) -> List[Dict[str, Any]]:
        if not DATA_DIR.exists():
            logger.warning("%s directory not found — no documents to ingest.", DATA_DIR)
            return []

        docs: List[Dict[str, Any]] = []
        for path in sorted(DATA_DIR.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            try:
                content = _extract_text(path).strip()
                if not content:
                    continue
                rel = str(path.relative_to(DATA_DIR))
                chunks = _chunk_text(content)
                for i, chunk in enumerate(chunks):
                    docs.append({
                        "id": f"{rel}::chunk{i}",
                        "content": chunk,
                        "metadata": {
                            "source": rel,
                            "file_name": path.name,
                            "file_suffix": path.suffix.lower(),
                            "chunk_index": i,
                            "chunk_total": len(chunks),
                        },
                    })
                logger.info("Read %s → %d chunk(s)", rel, len(chunks))
            except Exception as exc:
                logger.error("Failed to read %s: %s", path, exc)

        return docs

    def _ingest_data_dir(self) -> None:
        docs = self._read_files()
        if not docs:
            logger.info("No documents found in %s — collection unchanged.", DATA_DIR)
            return

        logger.info("Ingesting %d chunks from %s…", len(docs), DATA_DIR)
        self._db_client.delete_collection(name="rag_documents")
        self.collection = self._db_client.create_collection(
            name="rag_documents",
            metadata={"description": "Collection for RAG document storage"},
        )

        for doc in docs:
            try:
                self.collection.add(
                    documents=[doc["content"]],
                    metadatas=[doc["metadata"]],
                    ids=[doc["id"]],
                )
                logger.info("Ingested: %s", doc["id"])
            except Exception as exc:
                logger.error("Failed to ingest %s: %s", doc["id"], exc)

        logger.info(
            "Ingestion complete. Collection now has %d documents.",
            self.collection.count(),
        )

    def query(self, query: str, n_results: int) -> Dict[str, Any]:
        return self.collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )


# ---------------------------------------------------------------------------
# MCP tool — registered at decoration time, _server set at startup
# ---------------------------------------------------------------------------

@mcp.tool(name="search-document")
def search_document(query: str, n_results: int = 5) -> str:
    """
    Search the knowledge base for documents relevant to a query.

    Uses semantic vector similarity search via the bound embedding model
    to find the most relevant passages across all indexed documents.

    Args:
        query: The search query.
        n_results: Number of results to return (default: 5, max: 20).

    Returns:
        Formatted string with matching document passages and source references.
    """
    if _server is None:
        return "Server is still initialising — please retry in a moment."
    try:
        if not query.strip():
            return "Error: Query cannot be empty."

        n_results = max(1, min(n_results or 5, 20))
        results = _server.query(query, n_results)

        if not results["documents"] or not results["documents"][0]:
            return "No relevant documents found for your query."

        documents = results["documents"][0]
        metadatas = results["metadatas"][0] if results["metadatas"] else [{}] * len(documents)
        distances = results["distances"][0] if results["distances"] else [0.0] * len(documents)

        formatted = []
        for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances)):
            block = f"\n--- Result {i + 1} ---\nContent: {doc}\n"
            block += f"Source: {(meta or {}).get('source', 'Unknown')}\n"
            block += f"Similarity Score: {1 - dist:.3f}\n"
            formatted.append(block)

        logger.info("search-document '%s' returned %d results", query, len(documents))
        return (
            f"Found {len(documents)} relevant documents for query: '{query}'\n"
            + "\n".join(formatted)
        )
    except Exception as exc:
        logger.error("Error in search-document: %s", exc)
        return f"Error searching documents: {exc}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _redact_uri(uri: Optional[str]) -> Optional[str]:
    if not uri:
        return None
    try:
        parsed = urlparse(uri)
        if parsed.password:
            netloc = parsed.netloc.replace(parsed.password, "[REDACTED]")
            return urlunparse(parsed._replace(netloc=netloc))
    except Exception:
        pass
    return "[URI REDACTED]"


# ---------------------------------------------------------------------------
# Entry point — server initialised HERE, after tools are decorated
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    try:
        _server = RAGServer()
    except Exception as exc:
        logger.error("Failed to start RAG MCP Server: %s", exc)
        sys.exit(1)

    port = int(os.getenv("PORT", "8080"))
    logger.info("Starting RAG MCP Server on port %d…", port)
    mcp.run(transport="http", host="0.0.0.0", port=port)
