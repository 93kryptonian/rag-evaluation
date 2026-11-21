"""Answer generation and LLM-as-judge, using the OpenAI Responses API.

Needs OPENAI_API_KEY (put it in .env). Models can be overridden with
GENERATOR_MODEL / JUDGE_MODEL environment variables.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
client = OpenAI()  # reads OPENAI_API_KEY from the environment

GENERATOR_MODEL = os.getenv("GENERATOR_MODEL", "gpt-5.4-mini")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-5.5")

ANSWER_PROMPT = """Answer the customer's question using ONLY the help-center context below.
If the context does not contain the answer, reply exactly: "I don't know based on the help center."
Keep the answer to 1-3 sentences.

<context>
{context}
</context>

Question: {question}"""


def format_context(docs: list[dict]) -> str:
    return "\n\n".join(f"[{d['id']}] {d['title']}\n{d['text']}" for d in docs)


def generate_answer(question: str, docs: list[dict]) -> str:
    response = client.responses.create(
        model=GENERATOR_MODEL,
        input=ANSWER_PROMPT.format(context=format_context(docs), question=question),
    )
    return response.output_text.strip()


# ---------- LLM-as-judge ----------

class Verdict(BaseModel):
    reasoning: str  # written first, so the grade follows from the reasoning
    unsupported_claims: list[str]  # claims in the answer the context doesn't back up
    faithful: bool  # every claim in the answer is supported by the context
    answers_question: bool  # the answer addresses what was asked
    correct: Literal["correct", "partial", "incorrect"]  # vs the reference answer


JUDGE_PROMPT = """You are grading an answer produced by a RAG system.

<question>{question}</question>
<retrieved_context>
{context}
</retrieved_context>
<answer>{answer}</answer>
<reference_answer>{reference}</reference_answer>

Grade three things independently:
1. faithful: is EVERY factual claim in the answer supported by the retrieved context?
   List any unsupported claims. Saying "I don't know" is faithful.
2. answers_question: does the answer address the question that was asked?
3. correct: compared with the reference answer, is the answer correct, partial or incorrect?
   If the reference says the topic is not covered, an "I don't know" answer is correct."""


def judge(question: str, docs: list[dict], answer: str, reference: str) -> Verdict | None:
    response = client.responses.parse(
        model=JUDGE_MODEL,
        input=JUDGE_PROMPT.format(question=question, context=format_context(docs),
                                  answer=answer, reference=reference),
        text_format=Verdict,
    )
    return response.output_parsed  # None if the model refused or the output didn't parse
