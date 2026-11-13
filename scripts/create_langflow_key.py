"""Create a Langflow API key and store it in .env as LANGFLOW_API_KEY.

Needs Langflow running with LANGFLOW_AUTO_LOGIN=true (the local, single-user setup).
The key is written to .env only. It is not printed.

    python scripts/create_langflow_key.py [--url http://127.0.0.1:7860] [--env .env] [--force]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import httpx


def read_env(path: Path) -> dict[str, str]:
    values = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, _, value = line.partition("=")
                values[key.strip()] = value.strip()
    return values


def write_key(path: Path, key: str) -> None:
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    replaced = False
    for index, line in enumerate(lines):
        if line.split("=", 1)[0].strip() == "LANGFLOW_API_KEY":
            lines[index] = f"LANGFLOW_API_KEY={key}"
            replaced = True
    if not replaced:
        lines.append(f"LANGFLOW_API_KEY={key}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def key_works(url: str, key: str) -> bool:
    try:
        response = httpx.get(f"{url}/api/v1/flows/", params={"header_flows": "true"},
                             headers={"x-api-key": key}, timeout=15)
    except httpx.HTTPError:
        return False
    return response.status_code == 200


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://127.0.0.1:7860")
    parser.add_argument("--env", default=".env")
    parser.add_argument("--force", action="store_true", help="create a new key even if the stored one works")
    args = parser.parse_args()
    url = args.url.rstrip("/")
    env_path = Path(args.env)

    existing = read_env(env_path).get("LANGFLOW_API_KEY", "")
    if existing and not args.force and key_works(url, existing):
        print(f"LANGFLOW_API_KEY in {env_path} already works. Nothing to do.")
        return 0

    try:
        login = httpx.get(f"{url}/api/v1/auto_login", timeout=15)
    except httpx.HTTPError as exc:
        print(f"Could not reach Langflow at {url}: {exc}", file=sys.stderr)
        return 1
    if login.status_code != 200:
        print("Langflow did not allow auto-login. Start it with LANGFLOW_AUTO_LOGIN=true, or create a key in "
              "Langflow under Settings > Langflow API Keys and put it in .env yourself.", file=sys.stderr)
        return 1
    token = login.json()["access_token"]
    created = httpx.post(f"{url}/api/v1/api_key/", json={"name": "agent-configurator hand-off"},
                         headers={"Authorization": f"Bearer {token}"}, timeout=15)
    if created.status_code >= 300:
        print(f"Langflow refused to create a key (HTTP {created.status_code}).", file=sys.stderr)
        return 1
    write_key(env_path, created.json()["api_key"])
    print(f"Created a Langflow API key and saved it to {env_path}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
