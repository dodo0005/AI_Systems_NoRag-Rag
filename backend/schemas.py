from typing import Literal

from pydantic import BaseModel, Field, model_validator

RetrieverName = Literal["dense", "bm25", "graph"]


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=5000)
    document_ids: list[str] = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    corpus: Literal["vellora", "kestrel"] = "vellora"
    retrievers: list[RetrieverName] = Field(
        default_factory=lambda: ["dense", "bm25"],
        min_length=1,
    )

    @model_validator(mode="after")
    def validate_question_and_documents(self):
        self.question = self.question.strip()

        if not self.question:
            raise ValueError("Question cannot be blank.")

        self.document_ids = list(
            dict.fromkeys(
                document_id.strip()
                for document_id in self.document_ids
                if document_id.strip()
            )
        )

        if not self.document_ids:
            raise ValueError("Select at least one document.")

        return self


class Reference(BaseModel):
    document_id: str
    chunk_id: str
    text: str
    start_line: int
    end_line: int
    fused_rank: int | None = None
    found_by: dict[str, int] = Field(default_factory=dict)  # retriever -> rank


class Usage(BaseModel):
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0


class GraphEntity(BaseModel):
    name: str
    type: str


class GraphEdge(BaseModel):
    source: str
    relation: str
    target: str
    hop: int
    chunk_ids: list[str]
    quote: str


class GraphPath(BaseModel):
    entities: list[GraphEntity] = Field(default_factory=list)  # found in the question
    edges: list[GraphEdge] = Field(default_factory=list)  # followed in the walk
    chunk_ids: list[str] = Field(default_factory=list)  # the graph's ranked chunks
    in_context: list[str] = Field(default_factory=list)  # those that survived fusion


class AskResponse(BaseModel):
    status: Literal["answered", "insufficient_evidence"]
    answer: str
    references: list[Reference] = Field(default_factory=list)
    retrievers: list[str] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    latency_ms: dict[str, float] = Field(default_factory=dict)
    graph_path: GraphPath | None = None


class ErrorResponse(BaseModel):
    detail: str
