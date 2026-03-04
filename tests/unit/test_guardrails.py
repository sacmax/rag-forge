import pytest
from unittest.mock import AsyncMock
from rag_forge.guardrails.input_guard import InputGuard
from rag_forge.guardrails.relevance import RelevanceGate
from rag_forge.guardrails.output_guard import OutputGuard
from rag_forge.document.models import Chunk

def make_chunk(score: float) -> Chunk:
    return Chunk(id="c1", content="test", document_id="d1",
                 chunk_index=0, source="test.txt", metadata={"score": score})

# input Guard
def test_input_guard_rejects_empty():
    guard = InputGuard(max_length=500)
    is_valid, msg = guard.validate("")
    assert not is_valid

def test_input_guard_rejects_too_long():
    guard = InputGuard(max_length=10)
    is_valid, msg = guard.validate("a" * 50)
    assert not is_valid

def test_input_guard_accepts_valid():
    guard = InputGuard(max_length=500)
    is_valid, msg = guard.validate("What are the payment terms?")
    assert is_valid

# relevance gate
def test_relevance_gate_passes_high_score():
    gate = RelevanceGate(threshold=0.3)
    assert gate.check([make_chunk(0.8)])

def test_relevance_gate_fails_low_score():
    gate = RelevanceGate(threshold=0.3)
    assert not gate.check([make_chunk(0.1)])

def test_relevance_gate_filters():
    gate = RelevanceGate(threshold=0.5)
    chunks = [make_chunk(0.8), make_chunk(0.2), make_chunk(0.6)]
    filtered = gate.filter(chunks)
    assert len(filtered) == 2

# output guard
@pytest.mark.asyncio
async def test_output_guard_grounded():
    mock_llm = AsyncMock()
    mock_llm.complete = AsyncMock(return_value="GROUNDED")
    guard = OutputGuard(mock_llm, enable_scope_check=True)
    is_grounded, response = await guard.check("Answer text", [make_chunk(0.8)], "query")
    assert is_grounded

@pytest.mark.asyncio
async def test_output_guard_not_grounded():
    mock_llm = AsyncMock()
    mock_llm.complete = AsyncMock(return_value="NOT_GROUNDED: claim X is unsupported")
    guard = OutputGuard(mock_llm, enable_scope_check=True)
    is_grounded, response = await guard.check("Answer text", [make_chunk(0.8)], "query")
    assert not is_grounded

@pytest.mark.asyncio
async def test_output_guard_disabled():
    mock_llm = AsyncMock()
    guard = OutputGuard(mock_llm, enable_scope_check=False)
    is_grounded, response = await guard.check("Answer text", [make_chunk(0.8)], "query")
    assert is_grounded
    mock_llm.complete.assert_not_called()

    