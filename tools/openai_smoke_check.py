#!/usr/bin/env python3
"""Optional manual OpenAI API connectivity check.

This utility is not part of the automated ReplayBench-PG test suite.
"""

from __future__ import annotations

import os

from openai import OpenAI


def main() -> None:
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise SystemExit(
            "OPENAI_API_KEY is not set. "
            "Set it only when intentionally running this optional smoke check."
        )

    client = OpenAI(api_key=api_key)

    # A minimal authenticated request.
    models = client.models.list()
    print("[PASS] OpenAI API connection succeeded")
    print(f"[INFO] Models returned: {len(models.data)}")


if __name__ == "__main__":
    main()