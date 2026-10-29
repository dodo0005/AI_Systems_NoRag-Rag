from typing import Literal

from pydantic import BaseModel, Field, model_validator


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=5000)
    document_ids: list[str] = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)

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


class AskResponse(BaseModel):
    status: Literal["answered", "insufficient_evidence"]
    answer: str
    references: list[Reference] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    detail: str
