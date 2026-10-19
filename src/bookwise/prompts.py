"""Prompt templates kept separate from orchestration code."""

SUMMARY_PROMPTS: dict[str, str] = {
    "flash": """
You are an expert in concise communication.
Write a single dense, engaging paragraph of approximately 150-200 words about the
{document_type} below. Capture its central problem or question, key ideas, evidence
or solution, and practical significance.
Do not use bullet points.

Document text:
{text}
""",
    "chapters": """
You are an expert analyst.
Create a clean Markdown section-by-section summary of the {document_type} below.
For each major section include: Section Title, Key Idea, Analysis or Evidence, and Implications.
Be faithful to the provided text and do not invent details.

Document text:
{text}
""",
    "executive": """
You are a high-level consultant.
Write a professional executive summary of the {document_type} below, approximately 1000 words.
Use these sections: Executive Overview, Context & Audience, Key Ideas or Findings,
Evidence or Case Studies, Recommendations, and Conclusion.
Be faithful to the provided text and avoid unsupported claims.

Document text:
{text}
""",
}

TRANSLATION_PROMPT = """
Translate the following document summary into {language_name}.
Preserve the meaning, tone, paragraph structure, and Markdown formatting.
Do not add commentary or omit content.

Summary:
{text}
"""

QA_PROMPT = """
Create 20 useful question-and-answer pairs from the {document_type} below.
Focus on the most important ideas, definitions, arguments, examples, and practical lessons.
Use numbered Markdown entries. Be faithful to the provided text and do not invent details.

Document text:
{text}
"""

RAG_PROMPT = """
You are a careful document assistant.
Answer the question using only the supplied context. If the context does not contain
the answer, say that you do not know. Do not make up an answer.

Context:
{context}

Question:
{question}
"""
