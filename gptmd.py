#!/usr/bin/env python3

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect and export ChatGPT conversations to Markdown."
    )
    parser.add_argument(
        "-d",
        "--datapath",
        default="data",
        help="Folder containing conversations.json (default: data).",
    )
    parser.add_argument(
        "-l",
        "--list",
        action="store_true",
        help='List conversations as "YYYY-MM-DD - Title".',
    )
    parser.add_argument(
        "-s",
        "--stats",
        action="store_true",
        help="Show summary statistics for conversations.",
    )
    parser.add_argument(
        "-e",
        "--export",
        action="store_true",
        help="Export conversations to Markdown files.",
    )
    parser.add_argument(
        "-o",
        "--outdir",
        default="output",
        help="Destination directory for exports (default: output).",
    )
    parser.add_argument(
        "-t",
        "--include-thoughts",
        action="store_true",
        help="Include messages with content_type=thoughts.",
    )
    parser.add_argument(
        "-T",
        "--include-tools",
        action="store_true",
        help="Include assistant tool-call messages (content_type=code).",
    )
    parser.add_argument(
        "-u",
        "--utc",
        action="store_true",
        help="Format times in UTC instead of local time.",
    )
    return parser


def parse_args() -> Tuple[argparse.ArgumentParser, argparse.Namespace]:
    parser = build_parser()
    return parser, parser.parse_args()


def load_conversations(datapath: str) -> List[Dict[str, Any]]:
    conv_path = Path(datapath) / "conversations.json"
    if not conv_path.exists():
        raise FileNotFoundError(f"Could not find conversations.json at {conv_path}")
    with conv_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict):
        if isinstance(data.get("conversations"), list):
            return data["conversations"]
        if isinstance(data.get("items"), list):
            return data["items"]
        if data and all(isinstance(v, dict) for v in data.values()):
            return list(data.values())
        raise ValueError("Unexpected conversations.json format.")
    if isinstance(data, list):
        return data
    raise ValueError("Unexpected conversations.json format.")


def select_start_node(mapping: Dict[str, Dict[str, Any]], current_node: Optional[str]) -> Optional[str]:
    if current_node and current_node in mapping:
        return current_node
    latest_id = None
    latest_ts = None
    for node_id, node in mapping.items():
        msg = node.get("message") or {}
        ts = msg.get("update_time") or msg.get("create_time")
        if ts is None:
            continue
        if latest_ts is None or ts > latest_ts:
            latest_ts = ts
            latest_id = node_id
    if latest_id:
        return latest_id
    for node_id, node in mapping.items():
        if node.get("parent") is None:
            return node_id
    return next(iter(mapping), None)


def build_message_path(convo: Dict[str, Any]) -> List[str]:
    mapping = convo.get("mapping") or {}
    if not mapping:
        return []
    start_id = select_start_node(mapping, convo.get("current_node"))
    path: List[str] = []
    seen = set()
    node_id = start_id
    while node_id and node_id not in seen:
        seen.add(node_id)
        path.append(node_id)
        node = mapping.get(node_id) or {}
        parent_id = node.get("parent")
        node_id = parent_id
    path.reverse()
    return path


def resolve_citation_block(refs: List[str], safe_urls: List[str], ref_map: Dict[str, str], url_iter) -> str:
    links: List[str] = []
    for ref in refs:
        url = ref_map.get(ref)
        if not url:
            try:
                url = next(url_iter)
            except StopIteration:
                url = None
            if url:
                ref_map[ref] = url
        if url:
            links.append(f"[source]({url})")
    return " ".join(links)


def sanitize_text(raw: str, safe_urls: List[str]) -> str:
    if not raw:
        return ""

    # Replace private-use citation/nav blocks with markdown links derived from safe_urls.
    ref_map: Dict[str, str] = {}
    url_iter = iter(safe_urls or [])

    def replace_block(match: re.Match) -> str:
        body = match.group(1)
        tokens = body.split("")
        if not tokens:
            return ""
        tag = tokens[0]
        rest = tokens[1:]
        refs = [t for t in rest if t]
        if tag == "cite":
            return resolve_citation_block(refs, safe_urls, ref_map, url_iter)
        if tag == "navlist":
            # Try to use first non-ref token as label; else fall back to generic.
            label = next((t for t in rest if not t.startswith("turn")), "link")
            url = None
            for ref in refs:
                url = ref_map.get(ref)
                if url:
                    break
            if not url:
                try:
                    url = next(url_iter)
                except StopIteration:
                    url = None
            if url:
                return f"[{label}]({url})"
        return ""

    cleaned = re.sub(r"(.*?)", replace_block, raw)
    # Drop remaining private-use characters.
    cleaned = re.sub(r"[\ue000-\uf8ff]", "", cleaned)
    return cleaned


def render_content(content: Dict[str, Any], safe_urls: List[str]) -> str:
    ctype = content.get("content_type")
    if ctype == "text":
        parts = content.get("parts") or []
        raw = "\n\n".join(str(p) for p in parts if p is not None)
        return sanitize_text(raw, safe_urls)
    if ctype == "reasoning_recap":
        return sanitize_text(str(content.get("content") or ""), safe_urls)
    if ctype == "user_editable_context":
        profile = content.get("user_profile") or ""
        instructions = content.get("user_instructions") or ""
        chunks = [chunk for chunk in (profile, instructions) if chunk]
        return sanitize_text("\n".join(chunks), safe_urls)
    if "parts" in content:
        raw = "\n\n".join(str(p) for p in content.get("parts", []) if p is not None)
        return sanitize_text(raw, safe_urls)
    raw_fallback = json.dumps(content, ensure_ascii=True, sort_keys=True)
    return sanitize_text(raw_fallback, safe_urls)


def extract_messages(
    convo: Dict[str, Any], include_thoughts: bool, include_tools: bool
) -> List[Dict[str, Any]]:
    mapping = convo.get("mapping") or {}
    ordered_ids = build_message_path(convo)
    messages: List[Dict[str, Any]] = []
    safe_urls = convo.get("safe_urls") or []
    for node_id in ordered_ids:
        node = mapping.get(node_id) or {}
        msg = node.get("message")
        if not msg:
            continue
        content = msg.get("content") or {}
        ctype = content.get("content_type")
        if ctype == "thoughts" and not include_thoughts:
            continue
        if ctype == "code" and not include_tools:
            continue
        text = render_content(content, safe_urls)
        if not text.strip():
            continue
        metadata = msg.get("metadata") or {}
        messages.append(
            {
                "role": (msg.get("author") or {}).get("role") or "unknown",
                "text": text,
                "content_type": ctype,
                "create_time": msg.get("create_time"),
                "model": metadata.get("model_slug") or metadata.get("default_model_slug"),
            }
        )
    return messages


def fmt_timestamp(ts: Optional[float], use_utc: bool, date_only: bool = False) -> str:
    if ts is None:
        return "unknown"
    tz = timezone.utc if use_utc else None
    dt = datetime.fromtimestamp(ts, tz=tz)
    return dt.strftime("%Y-%m-%d") if date_only else dt.strftime("%Y-%m-%d %H:%M:%S")


def slugify(title: str) -> str:
    # Keep spaces and most printable characters, but strip path separators/control chars.
    cleaned = re.sub(r"[\\/\0]+", " ", title)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return "Untitled"
    return cleaned


def compute_stats(
    convos: Iterable[Dict[str, Any]], include_thoughts: bool, include_tools: bool
) -> Dict[str, Any]:
    convos = list(convos)
    role_counts: Dict[str, int] = defaultdict(int)
    total_messages = 0
    times: List[float] = []
    for convo in convos:
        if convo.get("create_time"):
            times.append(convo["create_time"])
        messages = extract_messages(convo, include_thoughts, include_tools)
        for msg in messages:
            total_messages += 1
            role_counts[msg["role"]] += 1
            if msg.get("create_time"):
                times.append(msg["create_time"])
    date_range: Tuple[Optional[float], Optional[float]] = (None, None)
    if times:
        date_range = (min(times), max(times))
    return {
        "conversation_count": len(convos),
        "total_messages": total_messages,
        "role_counts": dict(sorted(role_counts.items())),
        "date_range": date_range,
    }


def list_conversations(convos: Iterable[Dict[str, Any]], use_utc: bool) -> None:
    sorted_convos = sorted(
        convos,
        key=lambda c: c.get("create_time") or c.get("update_time") or 0,
    )
    for convo in sorted_convos:
        ts = convo.get("create_time") or convo.get("update_time")
        date_str = fmt_timestamp(ts, use_utc, date_only=True)
        title = convo.get("title") or "Untitled"
        print(f"{date_str} - {title}")


def write_export(
    convos: Iterable[Dict[str, Any]],
    outdir: str,
    include_thoughts: bool,
    use_utc: bool,
    include_tools: bool,
) -> None:
    Path(outdir).mkdir(parents=True, exist_ok=True)
    for convo in convos:
        messages = extract_messages(convo, include_thoughts, include_tools)
        title = convo.get("title") or "Untitled"
        ts = convo.get("create_time") or convo.get("update_time")
        date_str = fmt_timestamp(ts, use_utc, date_only=True)
        filename = f"{date_str} - {slugify(title)}.md"
        filepath = Path(outdir) / filename
        convo_id = convo.get("id") or convo.get("conversation_id") or "unknown"
        default_model = convo.get("default_model_slug")
        lines = [
            f"# {title}",
            "",
            f"- Created: {fmt_timestamp(convo.get('create_time'), use_utc)}",
            f"- Updated: {fmt_timestamp(convo.get('update_time'), use_utc)}",
            f"- Conversation ID: {convo_id}",
            f"- Link: https://chat.openai.com/c/{convo_id}",
            "",
        ]
        if default_model:
            lines.insert(4, f"- Default model: {default_model}")
        for msg in messages:
            time_label = fmt_timestamp(msg.get("create_time"), use_utc)
            role = msg.get("role") or "unknown"
            model_note = f" [{msg['model']}]" if msg.get("model") else ""
            body = msg.get("text") or ""
            lines.append(f"## {role}{model_note} @ {time_label}")
            lines.append("")
            lines.append(body)
            lines.append("")
        with filepath.open("w", encoding="utf-8") as fh:
            fh.write("\n".join(lines).rstrip() + "\n")


def main() -> None:
    parser, args = parse_args()

    if not (args.list or args.export or args.stats):
        parser.print_help()
        sys.exit(0)

    try:
        conversations = load_conversations(args.datapath)
    except (OSError, ValueError) as exc:
        print(f"Error loading conversations: {exc}", file=sys.stderr)
        sys.exit(1)

    did_action = False
    if args.list:
        list_conversations(conversations, args.utc)
        did_action = True
    if args.export:
        write_export(conversations, args.outdir, args.include_thoughts, args.utc, args.include_tools)
        print(f"Exported conversations to {os.path.abspath(args.outdir)}")
        did_action = True
    if args.stats:
        stats = compute_stats(conversations, args.include_thoughts, args.include_tools)
        date_range = stats["date_range"]
        start = fmt_timestamp(date_range[0], args.utc, date_only=True) if date_range[0] else "unknown"
        end = fmt_timestamp(date_range[1], args.utc, date_only=True) if date_range[1] else "unknown"
        print(f"Conversations: {stats['conversation_count']}")
        print(f"Total messages: {stats['total_messages']}")
        print(f"Date range: {start} -> {end}")
        print("By role:")
        for role, count in stats["role_counts"].items():
            print(f"  {role}: {count}")
        did_action = True
    if not did_action:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
