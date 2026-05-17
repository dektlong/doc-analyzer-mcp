# Document Search MCP Server

A **Retrieval Augmented Generation (RAG)** MCP server built with [FastMCP](https://github.com/jlowin/fastmcp) and PostgreSQL + pgvector, deployed on **Tanzu Platform for Cloud Foundry**.

## How it Works

1. **Document Ingestion**: At startup the server reads every file in the `documents/` directory and extracts its text (PDF and common text formats supported).
2. **Vectorization**: Text is embedded via the GenAI embedding model bound through Tanzu's service marketplace.
3. **Storage**: Embeddings are persisted in a PostgreSQL database with the pgvector extension, also bound as a Tanzu service.
4. **Search**: The `search-document` MCP tool performs semantic similarity search against the stored embeddings and returns the most relevant passages.

## Prerequisites

- Tanzu Platform for Cloud Foundry access (`cf` CLI authenticated)
- A PostgreSQL service instance with the `pgvector` extension
- A GenAI service instance providing both an **embedding** model and a **chat** model

## Service Setup

```bash
# PostgreSQL with pgvector
cf create-service <postgres-tile> <plan> document-db

# GenAI — multi-model plan exposing embedding + chat endpoints
cf create-service genai <multi-model-plan> document-embedding
cf create-service genai <chat-plan>        document-chat

# MCP gateway (registers this app as an MCP server)
cf create-service <mcp-gateway-tile> <plan> dekt-mcp-gw1
```

The binding credentials are read automatically from `VCAP_SERVICES` at startup. No manual environment variable configuration is required.

## Adding Documents

Place files to be indexed in the `documents/` directory before pushing:

```
documents/
├── factory-maintenance-log.pdf
├── operations-manual.txt
└── ...
```

Supported formats: `.pdf` `.txt` `.md` `.rst` `.csv` `.json` `.yaml` `.yml` `.toml` `.xml` `.html` `.log` `.sql`

To re-index after adding files, redeploy with `cf push`.

## Deploy

```bash
cf push
```

The `Procfile` starts the server; dependencies are installed from the pre-vendored `vendor/` directory so no network access is required during staging.

## MCP Tool

| Tool | Description |
|---|---|
| `search-document` | Semantic search over all indexed documents. Pass the user's question as `query`; returns the most relevant passages. |

### Example call

```json
{
  "method": "tools/call",
  "params": {
    "name": "search-document",
    "arguments": {
      "query": "What is the maintenance interval for pump P-101?",
      "n_results": 5
    }
  }
}
```

## Project Structure

```
.
├── Procfile                  # CF start command
├── manifest.yml              # CF deployment manifest
├── requirements.txt          # Pinned Python dependencies
├── runtime.txt               # Python version for buildpack
├── vendor/                   # Pre-downloaded wheels (offline install)
├── documents/                # Source documents to index
└── src/
    ├── rag_server.py         # MCP server + tool definitions
    ├── pg_vector_store.py    # PostgreSQL + pgvector storage backend
    └── cf_config.py          # VCAP_SERVICES credential parser
```

## Configuration

The only optional environment variable is:

| Variable | Default | Purpose |
|---|---|---|
| `EMBEDDING_DIMENSION` | `1536` | Override vector dimension if your embedding model uses a different size |

All other configuration (database URI, model endpoints, API keys) is derived automatically from `VCAP_SERVICES`. If any required binding is missing the app will log the error and exit rather than start in a degraded state.

## Vendoring Dependencies

The `vendor/` directory contains pre-downloaded Linux x86_64 wheels so staging never hits the network. To refresh after changing `requirements.txt`:

```bash
pip3 download \
  --platform manylinux_2_17_x86_64 \
  --platform manylinux2014_x86_64 \
  --platform linux_x86_64 \
  --python-version 3.11 \
  --implementation cp \
  --only-binary=:all: \
  -r requirements.txt \
  -d vendor/
```
