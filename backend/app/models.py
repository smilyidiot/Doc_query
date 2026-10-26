"""Pydantic models for the DocQuery API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    k: int = Field(3, ge=1, le=10)


class Source(BaseModel):
    source: str
    score: float
    snippet: str


class Answer(BaseModel):
    question: str
    answer: str
    confidence: float
    sources: list[Source]
