#!/usr/bin/env python3
"""
app/chat.py
CampusGuide RAG — CLI entry point.

Run from the ElasticSearch/ root:
    python -m app.chat
or:
    python app/chat.py
"""

import os
import sys

from dotenv import load_dotenv
load_dotenv()

from app.router.router import DOMAIN_DOCUMENTS, DOMAIN_TUITION, DOMAIN_CALENDAR, DOMAIN_CONTACTS
from app.core.orchestrator import handle_turn
from app.common.llm_client import _gpt_credentials_configured

SEP = "─" * 64

# ── Debug printer ─────────────────────────────────────────────────────────────

def _print_router(domains: list):
    if not domains:
        print("  [ROUTER]  no domain matched (below confidence threshold)")
    else:
        print(f"  [ROUTER]  → {', '.join(domains)}")


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    if not _gpt_credentials_configured():
        print("[ERROR] Configure the selected LLM provider credentials in .env")
        sys.exit(1)

    print("CampusGuide RAG  (DOCUMENTS · TUITION · CALENDAR · CONTACTS)")
    print("Type 'exit' to quit.\n")

    history:        list = []
    tuition_state:  dict = {}
    calendar_state: dict = {}

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not query or query.lower() in {"exit", "quit"}:
            break

        print()

        reply, domains, tuition_state, calendar_state, is_clar = handle_turn(
            query, history, tuition_state, calendar_state, stream=False
        )

        _print_router(domains)
        print(f"\nAssistant: {reply}\n")
        print(SEP)

        history += [{"role": "user", "content": query}, {"role": "assistant", "content": reply}]


if __name__ == "__main__":
    main()
