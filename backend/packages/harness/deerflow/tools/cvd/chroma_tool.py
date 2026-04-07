"""CVD Chroma SOP Vector Query Tool for DeerFlow.

Query SOP documents from ChromaDB using semantic search.
"""

import os
from typing import Annotated, Literal

from langchain_core.runnables import RunnableConfig
from langchain.tools import InjectedToolCallId, ToolRuntime, tool

from deerflow.agents.thread_state import ThreadState

# Chroma config from environment
CHROMA_PATH = os.environ.get("CHROMA_PATH", "./data/chroma_sop_v3")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "herald/dmeta-embedding-zh:latest")


def _get_embedding_function():
    """Create Ollama embedding function."""
    try:
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_URL)
    except ImportError:
        # Fallback: use a simple dummy embedding for testing
        class DummyEmbeddings:
            def embed_query(self, text):
                return [0.0] * 768
            def embed_documents(self, texts):
                return [[0.0] * 768 for _ in texts]
        return DummyEmbeddings()


def _query_chroma(question: str, top_k: int = 5) -> list[dict]:
    """Query ChromaDB for relevant SOP chunks."""
    import chromadb

    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Try different collection names
    for col_name in ["cvd_sop", "sop_full", "film_anomaly"]:
        try:
            collection = client.get_collection(name=col_name)
            break
        except Exception:
            continue
    else:
        return []

    # Get embedding
    embed_fn = _get_embedding_function()
    try:
        query_embedding = embed_fn.embed_query(question)
    except Exception:
        return []

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    output = []
    if results and results.get("documents"):
        for i, (doc, metadata) in enumerate(zip(results["documents"][0], results.get("metadatas", [[{}]*len(results["documents"][0])])[0])):
            output.append({
                "content": doc[:500],  # Limit content length
                "source": metadata.get("source", "unknown"),
            })
    return output


@tool("chroma_query", parse_docstring=True)
def chroma_query_tool(
    runtime: ToolRuntime[Literal, ThreadState],
    question: Annotated[str, InjectedToolCallId],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> str:
    """Query SOP documents from ChromaDB vector store.

    Use this tool when the user asks about:
    - SOP procedures (standard operating procedures)
    - Document details or specifications
    - How to perform specific tasks
    - Technical documentation content

    This performs semantic search over SOP documents to find relevant passages.

    Args:
        question: The search query in natural language.
            Examples:
            - "CVD 设备日常维护步骤"
            - "如何进行腔体清洁"
            - "RF 匹配操作规程"
    """
    try:
        results = _query_chroma(question, top_k=5)

        if not results:
            return f"No SOP documents found for: {question}"

        output = f"Found {len(results)} relevant SOP passages:\n\n"
        for i, r in enumerate(results, 1):
            output += f"--- Passage {i} ---\n"
            output += f"Source: {r['source']}\n"
            output += f"Content: {r['content']}\n\n"

        return output

    except Exception as e:
        return f"ChromaDB query error: {str(e)}"
