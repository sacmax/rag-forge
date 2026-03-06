import pytest
import json
from unittest.mock import AsyncMock, MagicMock
from rag_forge.agent.router import QueryRouter, RouterStrategy
from rag_forge.agent.evaluator import AnswerEvaluator, Grade
from rag_forge.agent.executor import AgentExecutor
from rag_forge.document.models import Chunk

@pytest.fixture
def mock_llm():
    return AsyncMock()

@pytest.fixture
def mock_pipeline():
    pipeline = MagicMock()
    pipeline._retriever = AsyncMock()
    pipeline._retriever.search.return_value = [
        Chunk(id="c1", content="Some relevant content", document_id="doc1", chunk_index=0, source="test.pdf")
    ]
    pipeline._llm = AsyncMock()
    pipeline._llm.complete.return_value = "Final answer"
    return pipeline

# router test
@pytest.mark.asyncio
async def test_router_single_hop(mock_llm):
    # Arrange
    mock_llm.complete.return_value = json.dumps({"strategy": "SINGLE_HOP", "reason": "needs doc lookup"})
    router = QueryRouter(mock_llm)
    # Act
    result = await router.route("What are the payment terms?")
    # Assert
    assert result == RouterStrategy.SINGLE_HOP

@pytest.mark.asyncio
async def test_router_fallback_on_bad_json(mock_llm):
    # If LLM returns unparseable text, default to SINGLE_HOP
    mock_llm.complete.return_value = "not valid json at all"
    router = QueryRouter(mock_llm)
    result = await router.route("What is the capital of France?")
    assert result == RouterStrategy.SINGLE_HOP

# evaluator test
@pytest.mark.asyncio
async def test_evaluator_correct(mock_llm):
    mock_llm.complete.return_value = json.dumps({"grade": "CORRECT", "reason": "directly answers"})
    evaluator = AnswerEvaluator(mock_llm)
    chunks = [Chunk(id="c1", content="Payment is due in 30 days", document_id="d1", chunk_index=0, source="contract.pdf")]
    result = await evaluator.evaluate("When is payment due?", chunks)
    assert result == Grade.CORRECT

@pytest.mark.asyncio
async def test_evaluator_fallback_on_bad_json(mock_llm):
    mock_llm.complete.return_value = "oops"
    evaluator = AnswerEvaluator(mock_llm)
    result = await evaluator.evaluate("Any question", [])
    assert result == Grade.AMBIGUOUS

# executor test
@pytest.mark.asyncio
async def test_executor_direct_strategy(mock_llm, mock_pipeline):
    # Router returns DIRECT, executor skips retrieval entirely
    mock_llm.complete.side_effect = [
        json.dumps({"strategy": "DIRECT", "reason": "general knowledge"}),  # router call
        "Paris is the capital of France",                               # direct answer call
    ]
    executor = AgentExecutor(pipeline=mock_pipeline, llm=mock_llm, max_hops=3)
    result = await executor.run("What is the capital of France?")
    assert result.total_hops == 0
    assert result.final_grade == "DIRECT"
    assert result.steps == []

@pytest.mark.asyncio
async def test_executor_single_hop_correct(mock_llm, mock_pipeline):
    # Router returns SINGLE_HOP, evaluator returns CORRECT on first hop
    mock_llm.complete.side_effect = [
        json.dumps({"strategy": "SINGLE_HOP", "reason": "doc lookup"}),  # router
        json.dumps({"grade": "CORRECT", "reason": "found answer"}),       # evaluator hop 1
        "Payment is due in 30 days",                                       # direct answer
    ]
    executor = AgentExecutor(pipeline=mock_pipeline, llm=mock_llm, max_hops=3)
    result = await executor.run("What are the payment terms?")
    assert result.total_hops == 1
    assert result.final_grade == "CORRECT"
    assert result.steps[0].action == "answer"

@pytest.mark.asyncio
async def test_executor_refine_then_correct(mock_llm, mock_pipeline):
    # Hop 1: AMBIGUOUS, refine. Hop 2: CORRECT, answer
    mock_llm.complete.side_effect = [
        json.dumps({"strategy": "SINGLE_HOP", "reason": "doc lookup"}),   # router
        json.dumps({"grade": "AMBIGUOUS", "reason": "partial match"}),     # evaluator hop 1
        "payment terms clause",                                             # rewrite query
        json.dumps({"grade": "CORRECT", "reason": "found it"}),            # evaluator hop 2
        "Payment is due in 30 days",                                        # final answer
    ]
    executor = AgentExecutor(pipeline=mock_pipeline, llm=mock_llm, max_hops=3)
    result = await executor.run("What are the payment terms?")
    assert result.total_hops == 2
    assert result.steps[0].action == "refine"
    assert result.steps[1].action == "answer"
    assert result.final_grade == "CORRECT"