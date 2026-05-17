"""
Pydantic models for the SHL Assessment Recommender API.
Schema is non-negotiable — deviating breaks the automated evaluator.
"""
from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, field_validator


class Message(BaseModel):
    role: str   # "user" | "assistant"
    content: str

    @field_validator("role")
    @classmethod
    def role_must_be_valid(cls, v: str) -> str:
        if v not in ("user", "assistant", "system"):
            raise ValueError(f"role must be 'user' or 'assistant', got {v!r}")
        return v


class ChatRequest(BaseModel):
    messages: List[Message]

    @field_validator("messages")
    @classmethod
    def messages_not_empty(cls, v: List[Message]) -> List[Message]:
        if not v:
            raise ValueError("messages list must not be empty")
        return v


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str  # e.g. "K", "A,P", "B,S"


class ChatResponse(BaseModel):
    reply: str
    recommendations: List[Recommendation]   # empty list when gathering context
    end_of_conversation: bool