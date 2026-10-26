"""Ask the agent questions from the terminal.

  python chat.py "How many PTO days carry over?"
  python chat.py            # interactive mode
"""
import sys

from agent.core import Agent


def ask(agent: Agent, question: str) -> None:
    res = agent.run(question, verbose=True)
    print(f"\n{res.answer}\n")
    print(f"({res.turns} turns, {len(res.tool_calls)} tool calls, "
          f"{res.input_tokens + res.output_tokens:,} tokens, {res.latency_s:.1f}s)\n")


def main() -> None:
    agent = Agent()
    if len(sys.argv) > 1:
        ask(agent, " ".join(sys.argv[1:]))
        return
    print("Acme HR helpdesk. Ask a question (Ctrl-D to quit).")
    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if q:
            ask(agent, q)


if __name__ == "__main__":
    main()
