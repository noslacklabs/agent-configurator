"""Hide Langflow's own model blocks so a Configured LLM is the only model source.

Uses Langflow's catalog policy (PUT /api/v1/catalog-policy/components), which is
stored in Langflow's database. Blocked blocks disappear from the sidebar and
flows that contain them are refused. Other blocked keys already in the policy
are kept.

    python scripts/set_block_policy.py            block the default set
    python scripts/set_block_policy.py --show     print the current policy
    python scripts/set_block_policy.py --clear    unblock the default set again

Not blocked on purpose: blocks that merely *have* a model picker (Structured
Output, Smart Router, Batch Run and others). They also accept a connected
Configured LLM, and hiding them would remove features, not key entry points.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import httpx

# Langflow's Agent, Language Model and Embedding Model blocks: each exists to pick a
# provider and take its key. The CoworkAI Agent, Configured LLM and Configured
# Embeddings blocks replace them.
DEFAULT_BLOCKED = ("Agent", "LanguageModelComponent", "EmbeddingModel")


def read_env(path: Path) -> dict[str, str]:
    values = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, _, value = line.partition("=")
                values[key.strip()] = value.strip()
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default=None)
    parser.add_argument("--env", default=".env")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--clear", action="store_true")
    parser.add_argument("--block", nargs="*", default=None, help="block keys to use in place of the default set")
    args = parser.parse_args()

    env = read_env(Path(args.env))
    url = (args.url or env.get("LANGFLOW_URL") or "http://127.0.0.1:7860").rstrip("/")
    api_key = env.get("LANGFLOW_API_KEY", "")
    if not api_key:
        print("LANGFLOW_API_KEY is missing. Run scripts/create_langflow_key.py first.", file=sys.stderr)
        return 1
    ours = set(args.block if args.block is not None else DEFAULT_BLOCKED)

    with httpx.Client(base_url=url, headers={"x-api-key": api_key}, timeout=60) as langflow:
        current = langflow.get("/api/v1/catalog-policy/components")
        if current.status_code != 200:
            print(f"Could not read the policy (HTTP {current.status_code}). The key must belong to a superuser.",
                  file=sys.stderr)
            return 1
        body = current.json()
        blocked = set(body.get("blocked") or [])
        if args.show:
            print("Blocked blocks:", ", ".join(sorted(blocked)) or "(none)")
            return 0
        if body.get("managed_externally"):
            print("This Langflow's block policy is managed outside Langflow. Nothing changed.", file=sys.stderr)
            return 1
        wanted = blocked - ours if args.clear else blocked | ours
        if wanted == blocked:
            print("Block policy already up to date:", ", ".join(sorted(blocked)) or "(none)")
            return 0
        updated = langflow.put("/api/v1/catalog-policy/components", json={"blocked": sorted(wanted)})
        if updated.status_code != 200:
            print(f"Langflow refused the policy (HTTP {updated.status_code}): {updated.text[:300]}", file=sys.stderr)
            return 1
        print("Blocked blocks now:", ", ".join(sorted(updated.json().get("blocked") or [])) or "(none)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
