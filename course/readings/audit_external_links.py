#!/usr/bin/env python3
"""Audit HTTP reachability of Markdown links; this is not a scientific review."""

import argparse
import concurrent.futures
import datetime
import functools
import html
import json
import os
import re
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

ROOT = Path(__file__).resolve().parents[2]
READINGS = ROOT / "course/readings"
USER_AGENT = "Mozilla/5.0 (MLSys curriculum reference audit)"
SKIP = {
    ".git",
    ".obsidian",
    ".cache",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    ".nox",
    "venv",
    "env",
    "__pycache__",
    "build",
    "CMakeFiles",
    "out",
    "dist",
    "target",
    "_deps",
    "results",
    "artifacts",
    "external",
    "third_party",
    "third-party",
    "vendor",
    "node_modules",
}


def markdown_urls(text):
    """Read inline links, reference definitions and autolinks outside code."""
    prose = []
    fence = None
    for line in text.splitlines(keepends=True):
        marker = re.match(r"^\s*(?:>\s*)?(`{3,}|~{3,})(.*)$", line)
        if fence:
            if marker and marker[1][0] == fence[0] and len(marker[1]) >= len(fence):
                if not marker[2].strip():
                    fence = None
        elif marker:
            fence = marker[1]
        else:
            prose.append(line)
    text = re.sub(r"(`+)(?!`)([\s\S]*?)(?<!`)\1(?!`)", "", "".join(prose))

    urls = set()
    starts = list(re.finditer(r"\]\(\s*(?P<url><?https?://)", text))
    starts.extend(re.finditer(r"(?m)^ {0,3}\[[^\]\n]+\]:\s*(?P<url><?https?://)", text))
    for match in starts:
        start = match.start("url")
        if text[start] == "<":
            end = text.find(">", start + 1)
            if end == -1:
                continue
            url = text[start + 1 : end]
        else:
            end = start
            depth = 0
            while end < len(text) and not text[end].isspace():
                char = text[end]
                if char == "\\" and end + 1 < len(text):
                    end += 2
                    continue
                if char == "(":
                    depth += 1
                elif char == ")":
                    if depth == 0:
                        break
                    depth -= 1
                end += 1
            url = text[start:end]
        urls.add(html.unescape(re.sub(r"\\([\\()[\]<>])", r"\1", url)))
    urls.update(html.unescape(url) for url in re.findall(r"<(https?://[^\s<>]+)>", text))
    return urls


def collect_urls(root=ROOT):
    urls = {}
    for folder, directories, files in os.walk(root):
        directories[:] = sorted(
            name
            for name in directories
            if name not in SKIP and not name.startswith((".venv", "build-", "cmake-build-"))
        )
        for name in sorted(files):
            if not name.endswith(".md"):
                continue
            path = Path(folder) / name
            for url in sorted(markdown_urls(path.read_text(encoding="utf-8"))):
                urls.setdefault(url, []).append(path.relative_to(root).as_posix())
    return urls


def check(item, timeout=18):
    url, files = item
    result = {"url": url, "referenced_by": sorted(set(files))}
    try:
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response.read(1024)
            result.update(http_status=response.status, final_url=response.url, status="reachable")
    except urllib.error.HTTPError as error:
        result.update(
            http_status=error.code,
            final_url=error.url,
            status=(
                "restricted_or_rate_limited" if error.code in (401, 403, 429) else "http_error"
            ),
            error=str(error),
        )
        error.close()
    except Exception as error:
        result.update(status="network_error", error=str(error))
    return result


def client_timezone():
    try:
        return ZoneInfo("Asia/Shanghai")
    except ZoneInfoNotFoundError:
        # Windows may lack IANA timezone data; Shanghai currently uses UTC+08:00.
        return datetime.timezone(datetime.timedelta(hours=8))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=READINGS / "results/external-links.json")
    parser.add_argument(
        "--workers", type=int, default=6, help="Concurrent HTTP requests (default: 6)"
    )
    parser.add_argument(
        "--timeout", type=float, default=18, help="Socket timeout in seconds (default: 18)"
    )
    args = parser.parse_args(argv)
    if args.workers < 1 or args.timeout <= 0:
        parser.error("--workers and --timeout must be positive")

    urls = sorted(collect_urls().items())
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(functools.partial(check, timeout=args.timeout), urls):
            results.append(result)
            if len(results) % 25 == 0:
                print(f"Checked {len(results)}/{len(urls)} URLs", flush=True)

    now = datetime.datetime.now(datetime.timezone.utc)
    summary = dict(sorted(Counter(result["status"] for result in results).items()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "checked_client_date": now.astimezone(client_timezone()).date().isoformat(),
                "utc": now.isoformat(),
                "source_root": ".",
                "scope": "HTTP reachability only; HTTP errors do not prove missing content.",
                "summary": summary,
                "links": results,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print("URLs", len(results), "reachable", summary.get("reachable", 0))
    for result in results:
        if result["status"] != "reachable":
            print(result["status"], result.get("http_status"), result["url"])
    print(f"RECORD: {args.output}")


if __name__ == "__main__":
    main()
