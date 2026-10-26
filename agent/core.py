"""The agent loop: send messages, run any tool calls, repeat until the model answers."""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field

from .tools import TOOL_SCHEMAS, run_tool

DEFAULT_MODEL = os.environ.get("AGENT_MODEL", "claude-sonnet-5-5")

SYSTEM_PROMPT = """You are the Acme Robotics employee helpdesk. You answer questions using the company handbook.

Rules:
- For any question about the company or its policies, call search_docs first. Answer only from what it returns, never from general knowledge.
- Cite the chunk each fact came from, using its ID in square brackets right after the fact, e.g. "Hotels are capped at $250 per night [travel-booking#0]."
- Use the calculate tool for every calculation, even simple ones.
- Use get_current_date when the question depends on today's date.
- If the search results do not contain the answer, reply starting with "I don't know" and suggest asking the People team. Do not guess, and do not cite anything.
- Keep answers to 1-4 sentences."""

CITATION_RE = re.compile(r"\[([a-z0-9-]+#\d+)\]")


@dataclass
class ToolCall:
    name: str
    input: dict
    output: str
    is_error: bool


@dataclass
class AgentResult:
    question: str
    answer: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    turns: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    stopped_early: bool = False

    @property
    def citations(self) -> list[str]:
        return sorted(set(CITATION_RE.findall(self.answer)))

    @property
    def retrieved_ids(self) -> list[str]:
        ids: list[str] = []
        for call in self.tool_calls:
            if call.name == "search_docs":
                ids += [i for i in CITATION_RE.findall(call.output) if i not in ids]
        return ids


class Agent:
    def __init__(self, model: str = DEFAULT_MODEL, max_turns: int = 8, client=None):
        if client is None:
            import anthropic  # imported lazily so retrieval works without the SDK
            client = anthropic.Anthropic()
        self.client = client
        self.model = model
        self.max_turns = max_turns

    def run(self, question: str, verbose: bool = False) -> AgentResult:
        result = AgentResult(question=question, answer="")
        messages = [{"role": "user", "content": question}]
        start = time.perf_counter()

        for _ in range(self.max_turns):
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                tools=TOOL_SCHEMAS,
                messages=messages,
            )
            result.turns += 1
            result.input_tokens += resp.usage.input_tokens
            result.output_tokens += resp.usage.output_tokens
            messages.append({"role": "assistant", "content": resp.content})

            if resp.stop_reason != "tool_use":
                result.answer = "".join(b.text for b in resp.content if b.type == "text").strip()
                break

            tool_results = []
            for block in resp.content:
                if block.type != "tool_use":
                    continue
                output, is_error = run_tool(block.name, block.input)
                result.tool_calls.append(ToolCall(block.name, block.input, output, is_error))
                if verbose:
                    preview = output if len(output) < 200 else output[:200] + "..."
                    print(f"  -> {block.name}({block.input})\n     {preview}")
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": output,
                    "is_error": is_error,
                })
            messages.append({"role": "user", "content": tool_results})
        else:
            result.stopped_early = True
            result.answer = "(agent hit the turn limit without answering)"

        result.latency_s = time.perf_counter() - start
        return result
