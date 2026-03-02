from jinja2 import Template
from rag_forge.document.models import Chunk

RAG_PROMPT_TEMPLATE = """\
You are an expert assistant for business and legal documents.
Answer the question using ONLY the provided context.
If the answer is not in the context, respond: "I cannot find this information in the provided documents."
After each factual claim, include an inline citation: [Source: <filename>, Page: <page or N/A>]

Context:
{% for chunk in chunks %}
--- [{{ loop.index }}]{{ chunk.source }}, Page {{ chunk.page or 'N/A' }} ---
{{ chunk.content}}

{% endfor %}
Question: {{ query }}

Answer:
"""

def build_rag_prompt(query: str, chunks: list[Chunk]) -> str:
    return Template(RAG_PROMPT_TEMPLATE).render(query=query, chunks=chunks)