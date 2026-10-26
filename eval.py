"""Score the docs agent on evals/cases.jsonl.

  python eval.py --retrieval-only                      # retrieval hit@4 only; no Claude API calls
  python eval.py                                       # full agent run
  python eval.py --judge                               # + LLM-as-judge score
  python eval.py --label baseline                      # save results/baseline.{json,md}
  python eval.py --compare baseline chunk-150          # before/after table from two saved runs

Retrieval settings (also work with --retrieval-only):
  --embedder local|voyage|tfidf   --chunk-size 500   --overlap 50   --title-prefix   --naive-chunking

Metrics
  retrieval_hit   every evidence phrase for the case is inside one of the top-4 chunks
                  (--retrieval-only: the question is the query; full run: any search the agent made)
  hit_at_1, mrr   --retrieval-only: is the evidence in the #1 chunk; mean reciprocal rank of the
                  position where all evidence is first covered. Rank-sensitive, so they still move
                  when hit@4 is saturated on a small corpus.
  tool_choice     the agent called every expected tool
  answer_correct  answerable cases: every expected fact is in the answer ("a|b" = either)
                  and no forbidden string is
  refusal         unanswerable cases: the agent said it doesn't know, and invented nothing
  false_refusal   answerable cases where the agent wrongly said it doesn't know (lower is better)
  citations_valid every [chunk-id] cited was really returned by search_docs
  judge           optional: an LLM grades the answer 1-5 against the evidence, rescaled to 0-1
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path

from agent import tools
from agent.retrieval import DEFAULT_CHUNK_SIZE, DEFAULT_EMBEDDER, DEFAULT_OVERLAP

ROOT = Path(__file__).resolve().parent
CASES = ROOT / "evals" / "cases.jsonl"
RESULTS = ROOT / "results"

REFUSAL_PHRASES = [
    "i don't know", "i do not know", "not covered", "doesn't cover", "does not cover",
    "isn't covered", "not in the handbook", "couldn't find", "could not find", "no information",
    "doesn't mention", "does not mention", "doesn't say", "does not say", "not mentioned",
]

METRICS = ["retrieval_hit", "hit_at_1", "mrr", "tool_choice", "answer_correct", "refusal", "false_refusal",
           "citations_valid", "judge"]


def load_cases() -> list[dict]:
    return [json.loads(line) for line in CASES.read_text().splitlines() if line.strip()]


def norm(s: str) -> str:
    return " ".join(s.lower().replace("’", "'").split())


def expand(fact: str) -> str:
    """Fill in facts that depend on when the eval runs."""
    return fact.replace("{today_weekday}", f"{date.today():%A}")


# ------------------------------------------------------------ checks

def retrieval_hit(case: dict, chunk_texts: list[str]) -> float | None:
    evidence = case.get("evidence", [])
    if not evidence:
        return None
    blob = norm(" ".join(chunk_texts))
    return 1.0 if all(norm(e) in blob for e in evidence) else 0.0


def is_refusal(answer: str) -> bool:
    return any(p in norm(answer) for p in REFUSAL_PHRASES)


def answer_correct(case: dict, answer: str) -> tuple[float, list[str]]:
    text, problems = norm(answer), []
    for fact in case.get("expected_facts", []):
        if not any(norm(expand(alt)) in text for alt in fact.split("|")):
            problems.append(f"missing fact '{expand(fact)}'")
    for bad in case.get("forbidden", []):
        if norm(bad) in text:
            problems.append(f"contains forbidden '{bad}'")
    return (0.0 if problems else 1.0), problems


def tool_choice(case: dict, called: list[str]) -> tuple[float, list[str]]:
    missing = [t for t in case.get("expected_tools", []) if t not in called]
    return (0.0 if missing else 1.0), [f"didn't call {t}" for t in missing]


JUDGE_PROMPT = """You are grading a company helpdesk assistant. The handbook evidence below is the ground truth.

<evidence>
{evidence}
</evidence>
<question>{question}</question>
<answer>{answer}</answer>

{rule}
Score 1-5: 5 = correct and complete; 3 = partly correct or missing something important; 1 = wrong or made up.
Reply with JSON only: {{"score": <1-5>, "reason": "<one sentence>"}}"""


def judge(client, model: str, case: dict, answer: str) -> tuple[float, str]:
    answerable = case.get("answerable", True)
    rule = ("The handbook contains the answer." if answerable else
            "The handbook does NOT contain the answer. The correct response says so; any specific claim is made up.")
    prompt = JUDGE_PROMPT.format(evidence="\n".join(case.get("evidence", [])) or "(none)",
                                 question=case["question"], answer=answer, rule=rule)
    resp = client.messages.create(model=model, max_tokens=200, messages=[{"role": "user", "content": prompt}])
    raw = "".join(b.text for b in resp.content if b.type == "text")
    try:
        data = json.loads(raw[raw.index("{"): raw.rindex("}") + 1])
        return (int(data["score"]) - 1) / 4, data.get("reason", "")
    except (ValueError, KeyError):
        return 0.0, f"unparseable judge reply: {raw[:80]}"


# ------------------------------------------------------------ runs

def run_retrieval_only(cases: list[dict]) -> list[dict]:
    index = tools.get_index()
    rows = []
    for case in cases:
        if not case.get("evidence"):
            continue
        hits = index.search(case["question"], k=tools.TOP_K)
        hit = retrieval_hit(case, [c.text for c, _ in hits])
        # Rank = how many top chunks you must read before all evidence is covered.
        rank = next((i for i in range(1, len(hits) + 1)
                     if retrieval_hit(case, [c.text for c, _ in hits[:i]])), None)
        rows.append({"id": case["id"], "category": case["category"],
                     "metrics": {"retrieval_hit": hit, "hit_at_1": 1.0 if rank == 1 else 0.0,
                                 "mrr": 1 / rank if rank else 0.0},
                     "notes": [] if hit else ["evidence not in top 4"],
                     "retrieved": [c.id for c, _ in hits]})
        print(f"  {'HIT ' if hit else 'MISS'}  {case['id']:<20} {[c.id for c, _ in hits]}")
    return rows


def run_agent(cases: list[dict], args) -> list[dict]:
    from agent.core import Agent

    agent = Agent(model=args.model)
    chunk_text = {c.id: c.text for c in tools.get_index().chunks}
    rows = []
    for case in cases:
        answerable = case.get("answerable", True)
        res = agent.run(case["question"], verbose=args.verbose)
        called = [c.name for c in res.tool_calls]
        retrieved = res.retrieved_ids
        m, notes = {}, []

        m["retrieval_hit"] = retrieval_hit(case, [chunk_text[i] for i in retrieved if i in chunk_text])
        if m["retrieval_hit"] == 0:
            notes.append("evidence not retrieved")
        m["tool_choice"], n = tool_choice(case, called); notes += n
        refused = is_refusal(res.answer)
        if answerable:
            m["answer_correct"], n = answer_correct(case, res.answer); notes += n
            m["false_refusal"] = 1.0 if refused else 0.0
        else:
            _, n = answer_correct(case, res.answer)  # forbidden strings only
            m["refusal"] = 1.0 if refused and not n else 0.0
            notes += n + ([] if refused else ["did not say it doesn't know"])
        bad_cites = [c for c in res.citations if c not in retrieved]
        m["citations_valid"] = 0.0 if bad_cites else 1.0
        notes += [f"cited {c} without retrieving it" for c in bad_cites]
        if args.judge:
            m["judge"], reason = judge(agent.client, args.judge_model, case, res.answer)
            if m["judge"] < 1:
                notes.append(f"judge: {reason}")
        if res.stopped_early:
            notes.append("hit the turn limit")

        passed = (m.get("answer_correct", m.get("refusal")) == 1 and m["tool_choice"] == 1
                  and m["citations_valid"] == 1)
        print(f"  {'PASS' if passed else 'FAIL'}  {case['id']:<20} {res.answer[:90]!r}")
        for note in notes:
            print(f"          - {note}")
        rows.append({"id": case["id"], "category": case["category"], "metrics": m, "notes": notes,
                     "answer": res.answer, "tools": called, "retrieved": retrieved,
                     "cited": res.citations, "turns": res.turns,
                     "tokens": res.input_tokens + res.output_tokens,
                     "latency_s": round(res.latency_s, 2)})
    return rows


# ------------------------------------------------------------ reporting

def mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def summarize(rows: list[dict]) -> dict:
    out = {}
    for metric in METRICS:
        vals = [r["metrics"].get(metric) for r in rows]
        n = len([v for v in vals if v is not None])
        if n:
            out[metric] = {"score": mean(vals), "n": n}
    if rows and "tokens" in rows[0]:
        out["avg_tokens"] = sum(r["tokens"] for r in rows) / len(rows)
        out["avg_latency_s"] = sum(r["latency_s"] for r in rows) / len(rows)
    return out


def fmt(v) -> str:
    return "  -  " if v is None else f"{v:.2f}"


def table(rows: list[dict]) -> str:
    present = [m for m in METRICS if any(r["metrics"].get(m) is not None for r in rows)]
    cats = sorted({r["category"] for r in rows}) + ["ALL"]
    lines = ["| category | n | " + " | ".join(present) + " |",
             "|---|---|" + "---|" * len(present)]
    for cat in cats:
        rs = rows if cat == "ALL" else [r for r in rows if r["category"] == cat]
        cells = [fmt(mean([r["metrics"].get(m) for r in rs])) for m in present]
        name = f"**{cat}**" if cat == "ALL" else cat
        lines.append(f"| {name} | {len(rs)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def save(label: str, config: dict, rows: list[dict]) -> None:
    RESULTS.mkdir(exist_ok=True)
    summary = summarize(rows)
    (RESULTS / f"{label}.json").write_text(json.dumps(
        {"label": label, "config": config, "summary": summary, "rows": rows}, indent=2))
    failures = [f"- `{r['id']}`: " + "; ".join(r["notes"]) for r in rows if r["notes"]]
    md = [f"# Eval run: {label}", "", "```", json.dumps(config, indent=2), "```", "", table(rows), ""]
    if "avg_tokens" in summary:
        md.append(f"Avg tokens/question: {summary['avg_tokens']:,.0f} · "
                  f"avg latency: {summary['avg_latency_s']:.1f}s\n")
    md += ["## Failures", ""] + (failures or ["None."])
    (RESULTS / f"{label}.md").write_text("\n".join(md) + "\n")
    print(f"\nSaved results/{label}.json and results/{label}.md")


def compare(before: str, after: str) -> None:
    a = json.loads((RESULTS / f"{before}.json").read_text())
    b = json.loads((RESULTS / f"{after}.json").read_text())
    print(f"| metric | {before} | {after} | change |\n|---|---|---|---|")
    for metric in METRICS:
        x, y = a["summary"].get(metric), b["summary"].get(metric)
        if x and y:
            d = y["score"] - x["score"]
            print(f"| {metric} | {x['score']:.2f} | {y['score']:.2f} | {d:+.2f} |")
    changed = {k: (a["config"].get(k), v) for k, v in b["config"].items()
               if a["config"].get(k) != v and k != "run_at"}
    print(f"\nConfig changes: {changed or 'none'}")
    ra = {r["id"]: r for r in a["rows"]}
    for r in b["rows"]:
        old = ra.get(r["id"])
        if old:
            for metric, v in r["metrics"].items():
                o = old["metrics"].get(metric)
                if o is not None and v is not None and o != v:
                    print(f"  {r['id']}: {metric} {o:.2f} -> {v:.2f}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--retrieval-only", action="store_true", help="score retrieval alone, no Claude API calls")
    p.add_argument("--embedder", default=DEFAULT_EMBEDDER, choices=["local", "voyage", "tfidf"])
    p.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    p.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    p.add_argument("--title-prefix", action="store_true", help="prepend the doc title to each chunk before embedding")
    p.add_argument("--naive-chunking", action="store_true",
                   help="let chunks end on a heading (the 'before' setting of the recorded experiment)")
    p.add_argument("--model", default=None, help="agent model (default: $AGENT_MODEL or claude-sonnet-5-5)")
    p.add_argument("--judge", action="store_true", help="add an LLM-as-judge score")
    p.add_argument("--judge-model", default=None, help="judge model (default: same as --model)")
    p.add_argument("--only", help="run only this category or case id")
    p.add_argument("--label", help="save results under results/<label>.{json,md}")
    p.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"), help="compare two saved runs")
    p.add_argument("-v", "--verbose", action="store_true", help="print every tool call")
    args = p.parse_args()

    if args.compare:
        compare(*args.compare)
        return 0

    tools.configure_index(embedder=args.embedder, chunk_size=args.chunk_size,
                          overlap=args.overlap, title_prefix=args.title_prefix,
                          heading_aware=not args.naive_chunking)
    cases = load_cases()
    if args.only:
        cases = [c for c in cases if args.only in (c["id"], c["category"])]
        if not cases:
            sys.exit(f"no cases match --only {args.only}")

    config = {"embedder": args.embedder, "chunk_size": args.chunk_size, "overlap": args.overlap,
              "title_prefix": args.title_prefix, "heading_aware": not args.naive_chunking,
              "top_k": tools.TOP_K,
              "run_at": datetime.now().isoformat(timespec="seconds")}
    if args.retrieval_only:
        config["mode"] = "retrieval-only"
        print(f"Retrieval-only: {args.embedder} embeddings, {args.chunk_size}-token chunks, top {tools.TOP_K}\n")
        rows = run_retrieval_only(cases)
    else:
        from agent.core import DEFAULT_MODEL
        args.model = args.model or DEFAULT_MODEL
        args.judge_model = args.judge_model or args.model
        config.update(mode="agent", model=args.model, judge=args.judge_model if args.judge else None)
        print(f"Agent run: {args.model}, {args.embedder} embeddings, {args.chunk_size}-token chunks\n")
        rows = run_agent(cases, args)

    print("\n" + table(rows))
    summary = summarize(rows)
    if "avg_tokens" in summary:
        print(f"\nAvg tokens/question: {summary['avg_tokens']:,.0f}   avg latency: {summary['avg_latency_s']:.1f}s")
    if args.label:
        save(args.label, config, rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
