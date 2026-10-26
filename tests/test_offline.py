"""Offline tests: no API keys, no model downloads.

  python -m pytest tests        or        python tests/test_offline.py
"""
import sys
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import tools  # noqa: E402
from agent.core import Agent  # noqa: E402
from agent.retrieval import VectorIndex, chunk_doc, count_tokens, load_chunks  # noqa: E402
from eval import answer_correct, is_refusal, retrieval_hit, tool_choice  # noqa: E402

tools.configure_index(embedder="tfidf")


# ---------------------------------------------------------------- fake API client

class FakeClient:
    """Replays a fixed script of model responses so the loop can be tested offline."""

    def __init__(self, script):
        self.script, self.calls = list(script), []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.script.pop(0)


def tool_use(id_, name, inp):
    return NS(type="tool_use", id=id_, name=name, input=inp)


def text(t):
    return NS(type="text", text=t)


def response(stop, *blocks):
    return NS(stop_reason=stop, content=list(blocks), usage=NS(input_tokens=100, output_tokens=20))


# ---------------------------------------------------------------- tests

def test_chunks_respect_size_and_keep_headings():
    for size in (500, 150):
        chunks = load_chunks(size, 50)
        assert max(count_tokens(c.text) for c in chunks) <= size
        assert not any(c.text.splitlines()[-1].startswith("#") for c in chunks)


def test_overlap_repeats_boundary_text():
    md = "# T\n" + " ".join(f"Sentence number {i} is here." for i in range(40))
    a, b = chunk_doc("t", md, chunk_size=60, overlap=15)[:2]
    assert a.text.splitlines()[-1] in b.text


def test_search_ranks_right_doc_first():
    index = VectorIndex("tfidf")
    top = index.search("hotel cap in New York", k=4)
    assert top[0][0].doc == "travel-booking"
    assert all(-1.0 <= s <= 1.0001 for _, s in top)


def test_calculate_is_safe():
    assert tools.calculate("350 * 4") == "1400"
    assert tools.calculate("$1,400 / 4") == "350"
    assert tools.calculate("__import__('os')").startswith("Error")
    assert tools.calculate("9 ** 9 ** 9").startswith("Error")
    assert tools.run_tool("nope", {})[1] is True


def test_agent_loop_runs_tools_and_records_trace():
    client = FakeClient([
        response("tool_use", tool_use("t1", "search_docs", {"query": "internet stipend"})),
        response("tool_use", tool_use("t2", "calculate", {"expression": "50 * 12"})),
        response("end_turn", text("You get $600 a year [remote-work#0].")),
    ])
    res = Agent(client=client).run("Annual internet stipend?")
    assert [c.name for c in res.tool_calls] == ["search_docs", "calculate"]
    assert res.tool_calls[1].output == "600"
    assert "remote-work#0" in res.retrieved_ids
    assert res.citations == ["remote-work#0"]
    # The tool result goes back with the matching tool_use_id.
    tool_msg = client.calls[-1]["messages"][-2]  # [-1] is the final assistant turn
    assert tool_msg["role"] == "user" and tool_msg["content"][0]["tool_use_id"] == "t2"


def test_scoring():
    case = {"expected_facts": ["$600|600"], "forbidden": ["%"], "evidence": ["$50 monthly internet stipend"],
            "expected_tools": ["search_docs", "calculate"]}
    assert answer_correct(case, "It's $600 a year")[0] == 1.0
    assert answer_correct(case, "It's 5% of salary")[0] == 0.0
    assert retrieval_hit(case, ["...a $50 monthly internet stipend, paid..."]) == 1.0
    assert retrieval_hit(case, ["unrelated"]) == 0.0
    assert tool_choice(case, ["search_docs"])[0] == 0.0
    assert is_refusal("I don’t know. Please ask the People team.")
    assert not is_refusal("Hotels are capped at $250 per night.")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
