from pydantic import BaseModel, ConfigDict, Field


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    query: str = Field(
        min_length=1,
        max_length=1000,
        description=(
            "User question for the RAG system. "
            "Must be between 1 and 1000 characters."
        ),
    )


class SourceDocument(BaseModel):
    id: str = Field(description="Source document identifier.")
    title: str | None = Field(
        default=None,
        description="Source document title, if available.",
    )
    content: str = Field(description="Retrieved source content.")


class QueryResponse(BaseModel):
    answer: str = Field(description="Generated answer from the RAG system.")
    sources: list[SourceDocument] = Field(
        description="Documents used as context for the generated answer."
    )
    model: str = Field(description="Generation model used for the answer.")
