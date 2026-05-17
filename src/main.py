import logging
import os

from rag_server import mcp  # importing rag_server creates _server and registers all tools

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    logger.info("Starting RAG MCP Server on port %d…", port)
    mcp.run(transport="http", host="0.0.0.0", port=port)
