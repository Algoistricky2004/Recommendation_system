"""
agent.py — Core agent logic for the SHL Assessment Recommender.

Uses Groq API (llama-3.1-8b-instant) for LLM inference.
CPU-friendly: inference is remote, retrieval is BM25 (pure Python).
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import List, Optional

from groq import Groq

from app.catalog import CatalogIndex, CatalogItem, get_catalog
from app.models import ChatRequest, ChatResponse, Message, Recommendation
from app.prompts import build_system_prompt
from app.retrieval import retrieve_candidates

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Groq client (singleton)
# ---------------------------------------------------------------------------
_groq_client: Optional[Groq] = None


def get_groq_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY environment variable is not set.")
        _groq_client = Groq(api_key=api_key)
    return _groq_client


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------
_MODEL = "llama-3.1-8b-instant"
_MAX_TOKENS = 1024
_TEMPERATURE = 0.15   # low for grounded, consistent recommendations


def _call_llm(system_prompt: str, messages: List[Message]) -> str:
    """
    Call Groq and return the raw text response.
    Raises on network/API errors.
    """
    client = get_groq_client()

    # Convert app Message objects to dicts expected by Groq
    groq_messages = [
        {"role": m.role, "content": m.content}
        for m in messages
        if m.role in ("user", "assistant")  # filter out any system role from history
    ]

    # Groq doesn't take system= kwarg; send as first system message in messages array
    groq_messages_with_sys = [
        {"role": "system", "content": system_prompt},
        *groq_messages,
    ]

    response = client.chat.completions.create(
        model=_MODEL,
        messages=groq_messages_with_sys,
        temperature=_TEMPERATURE,
        max_tokens=_MAX_TOKENS,
        top_p=1,
        stream=False,
        stop=None,
    )

    return response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# JSON parsing with robust fallback
# ---------------------------------------------------------------------------
def _extract_json(text: str) -> dict:
    """
    Extract JSON from LLM output.
    Handles: bare JSON, JSON inside ```json ... ```, partial wrapping.
    """
    text = text.strip()

    # Remove markdown code fences
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"```\s*$", "", text, flags=re.MULTILINE)
    text = text.strip()

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find the first { ... } block
    match = re.search(r"\{[\s\S]+\}", text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # If all fails, return a safe fallback
    logger.warning("Failed to parse LLM JSON output: %r", text[:300])
    return {
        "reply": text if text else "I'm sorry, I encountered an issue. Could you please rephrase?",
        "recommendations": [],
        "end_of_conversation": False,
    }


# ---------------------------------------------------------------------------
# Response validation and sanitisation
# ---------------------------------------------------------------------------
def _validate_and_sanitise(
    raw: dict,
    catalog: CatalogIndex,
) -> ChatResponse:
    """
    Validate the parsed LLM response against the catalog.
    - Verify every recommended item exists in the catalog.
    - Clamp recommendations to 1–10 (or 0 if empty).
    - Ensure URLs are catalog URLs.
    """
    reply: str = str(raw.get("reply", "")).strip()
    if not reply:
        reply = "I need a bit more information to make a good recommendation. Could you tell me more about the role?"

    end_of_conversation: bool = bool(raw.get("end_of_conversation", False))

    raw_recs = raw.get("recommendations", []) or []
    validated_recs: list[Recommendation] = []

    for rec in raw_recs:
        if not isinstance(rec, dict):
            continue
        name = str(rec.get("name", "")).strip()
        url = str(rec.get("url", "")).strip()
        test_type = str(rec.get("test_type", "")).strip()

        if not name or not url:
            continue

        # Verify the item exists in the catalog
        catalog_item = catalog.get_by_name(name)
        if catalog_item is None:
            # Try partial match
            catalog_item = _fuzzy_lookup(name, catalog)

        if catalog_item is None:
            logger.warning("LLM hallucinated product %r — dropping from shortlist", name)
            continue

        # Use catalog URL (never LLM-invented URL)
        validated_recs.append(Recommendation(
            name=catalog_item.name,
            url=catalog_item.url,
            test_type=test_type or catalog_item.test_type,
        ))

    # Clamp to 10
    validated_recs = validated_recs[:10]

    return ChatResponse(
        reply=reply,
        recommendations=validated_recs,
        end_of_conversation=end_of_conversation,
    )


def _fuzzy_lookup(name: str, catalog: CatalogIndex) -> Optional[CatalogItem]:
    """Partial / case-insensitive name matching as fallback."""
    nl = name.lower().strip()
    for item in catalog.items:
        if nl in item.name.lower() or item.name.lower() in nl:
            return item
    return None


# ---------------------------------------------------------------------------
# Turn-count guard
# ---------------------------------------------------------------------------
_MAX_TURNS = 8  # per the spec


def _count_turns(messages: List[Message]) -> int:
    return len(messages)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def process_chat(request: ChatRequest, catalog: Optional[CatalogIndex] = None) -> ChatResponse:
    """
    Main entry point.
    1. Retrieve relevant catalog candidates via BM25.
    2. Build system prompt with those candidates.
    3. Call Groq LLM with conversation history.
    4. Parse, validate, sanitise the response.
    5. Return structured ChatResponse.
    """
    if catalog is None:
        catalog = get_catalog()

    messages = request.messages

    # Guard: if turn cap reached, force a polite close
    if _count_turns(messages) > _MAX_TURNS:
        last_recs = _get_last_recommendations(messages, catalog)
        return ChatResponse(
            reply=(
                "We've reached the turn limit for this session. "
                "Here is the current shortlist. Please reach out to SHL for further assistance."
            ),
            recommendations=last_recs,
            end_of_conversation=True,
        )

    # Retrieve candidates
    candidates = retrieve_candidates(messages, catalog, top_k=28)

    # Build prompt
    system_prompt = build_system_prompt(candidates)

    # Call LLM
    raw_text = _call_llm(system_prompt, messages)

    # Parse
    raw_dict = _extract_json(raw_text)

    # Validate + sanitise
    response = _validate_and_sanitise(raw_dict, catalog)

    return response


def _get_last_recommendations(
    messages: List[Message],
    catalog: CatalogIndex,
) -> list[Recommendation]:
    """
    Extract the most recent shortlist from the assistant messages.
    Used when the turn cap is hit — we re-surface the last shortlist.
    """
    for msg in reversed(messages):
        if msg.role == "assistant":
            # Try to parse the assistant's last JSON response
            try:
                data = _extract_json(msg.content)
                recs_raw = data.get("recommendations", []) or []
                if recs_raw:
                    response = _validate_and_sanitise(data, catalog)
                    return response.recommendations
            except Exception:
                pass
    return []