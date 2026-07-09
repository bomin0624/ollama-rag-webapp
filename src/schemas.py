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
