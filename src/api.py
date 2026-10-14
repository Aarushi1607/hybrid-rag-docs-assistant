
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.generation import ask
from src.retrieval import load_retrieval_components


# Load retrieval components once when the API starts.
@asynccontextmanager
async def lifespan(app: FastAPI):
    vectorstore, bm25, chunks = load_retrieval_components()

    app.state.vectorstore = vectorstore
    app.state.bm25 = bm25
    app.state.chunks = chunks

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
        )
        return result

    except HTTPException:
        raise

    except Exception as exc:
        # Keep implementation details out of the API response.
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing your question.",
        ) from exc


@app.get("/v1/health")
def health():
    return {
        "status": "ok",
        "chunks_loaded": len(app.state.chunks),
    }
