import logging
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.generation import ask, warm_up, OLLAMA_MODEL
from src.retrieval import load_retrieval_components

logger = logging.getLogger("uvicorn.error")


# Load retrieval components once when the API starts.
@asynccontextmanager
async def lifespan(app: FastAPI):
    vectorstore, bm25, chunks = load_retrieval_components()

    app.state.vectorstore = vectorstore
    app.state.bm25 = bm25
    app.state.chunks = chunks

    # Load the LLM into memory now so the first question isn't slow.
    # The API still starts if Ollama is down; /v1/ask will report the error.
    try:
        warm_up()
    except Exception:
        logger.warning("Ollama warm-up failed (is Ollama running?)", exc_info=True)

    yield


app = FastAPI(
    title="Hybrid RAG Documentation Assistant",
    description="Ask questions about FastAPI documentation using RAG.",
    version="1.0.0",
    lifespan=lifespan,
)


class Question(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Question about the FastAPI documentation",
    )
    retrieval_mode: Literal["hybrid", "dense"] = Field(
        "hybrid",
        description="hybrid = Dense + BM25 + RRF (default); dense = vector search only",
    )


@app.post("/v1/ask")
def ask_question(body: Question):
    question = body.question.strip()

    if not question:
        raise HTTPException(
            status_code=422,
            detail="Question cannot be empty or contain only whitespace.",
        )

    try:
        result = ask(
            question,
            app.state.vectorstore,
            app.state.bm25,
            app.state.chunks,
            mode=body.retrieval_mode,
        )
        return result

    except HTTPException:
        raise

    except Exception as exc:
        # Log the cause server-side; keep implementation details out of the response.
        logger.exception("Failed to answer question (mode=%s)", body.retrieval_mode)
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing your question.",
        ) from exc


@app.get("/v1/health")
def health():
    return {
        "status": "ok",
        "chunks_loaded": len(app.state.chunks),
        "model": OLLAMA_MODEL,
    }