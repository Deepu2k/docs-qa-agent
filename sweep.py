"""Retrieval-only parameter sweep: chunk size x overlap x heading-aware chunking.

  python sweep.py                      # local embeddings (default)
  python sweep.py --embedder tfidf     # no download needed
Writes results/sweep-<embedder>.md. Makes no Claude API calls (voyage makes embedding calls).
"""
import argparse
import itertools

from agent.retrieval import VectorIndex
from eval import RESULTS, load_cases, mean, retrieval_hit


def score(index: VectorIndex, cases: list[dict]) -> tuple[float, float, float]:
    hit4, hit1, rr = [], [], []
    for case in cases:
        hits = index.search(case["question"], k=4)
        rank = next((i for i in range(1, len(hits) + 1)
                     if retrieval_hit(case, [c.text for c, _ in hits[:i]])), None)
        hit4.append(1.0 if rank else 0.0)
        hit1.append(1.0 if rank == 1 else 0.0)
        rr.append(1 / rank if rank else 0.0)
    return mean(hit4), mean(hit1), mean(rr)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--embedder", default="local", choices=["local", "voyage", "tfidf"])
    p.add_argument("--sizes", default="500,300,150,100")
    p.add_argument("--overlaps", default="0,50")
    args = p.parse_args()

    cases = [c for c in load_cases() if c.get("evidence")]
    lines = [f"# Retrieval sweep ({args.embedder} embeddings, top 4, {len(cases)} answerable cases)", "",
             "| chunk size | overlap | heading-aware | chunks | hit@4 | hit@1 | MRR |",
             "|---|---|---|---|---|---|---|"]
    sizes = [int(s) for s in args.sizes.split(",")]
    overlaps = [int(o) for o in args.overlaps.split(",")]
    for size, overlap, aware in itertools.product(sizes, overlaps, [False, True]):
        index = VectorIndex(args.embedder, size, overlap, heading_aware=aware)
        h4, h1, mrr = score(index, cases)
        line = (f"| {size} | {overlap} | {'yes' if aware else 'no'} | {len(index.chunks)} "
                f"| {h4:.2f} | {h1:.2f} | {mrr:.2f} |")
        print(line)
        lines.append(line)
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"sweep-{args.embedder}.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"\nSaved {out.relative_to(RESULTS.parent)}")


if __name__ == "__main__":
    main()
