from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS


# ----------------------------- Tool Implementations -----------------------------


def tool_web_search(query: str, max_results: int = 5, region: str = "wt-wt") -> Dict[str, Any]:
	"""Search the web using DuckDuckGo and return summarized results."""
	max_results = max(1, min(int(max_results), 10))
	results: List[Dict[str, Any]] = []
	with DDGS() as ddgs:
		for r in ddgs.text(query, region=region, max_results=max_results):
			results.append({
				"title": r.get("title") or "",
				"href": r.get("href") or r.get("url") or "",
				"snippet": r.get("body") or r.get("excerpt") or "",
			})
	if not results:
		return {"ok": True, "results": [], "summary": "No results found."}
	lines: List[str] = []
	for idx, item in enumerate(results, start=1):
		title = item["title"].strip()
		url = item["href"].strip()
		snippet = (item["snippet"] or "").strip()
		lines.append(f"{idx}. {title}\n{url}\n{snippet}")
	text = "\n\n".join(lines)
	return {"ok": True, "results": results, "summary": text}


def _clean_html_to_text(html: str) -> str:
	soup = BeautifulSoup(html, "html.parser")
	for tag in soup(["script", "style", "noscript"]):
		tag.decompose()
	text = soup.get_text(" ")
	text = re.sub(r"\s+", " ", text).strip()
	return text


def tool_web_fetch(url: str, timeout_seconds: int = 20, max_chars: int = 4000) -> Dict[str, Any]:
	"""Fetch and extract readable text content from a webpage."""
	resp = requests.get(url, timeout=timeout_seconds, headers={
		"User-Agent": "AgentBot/1.0 (+https://example.com)"
	})
	resp.raise_for_status()
	text = _clean_html_to_text(resp.text)
	if len(text) > max_chars:
		text = text[: max_chars - 3] + "..."
	return {"ok": True, "url": url, "content": text}


# ----------------------------- Tool Registry -----------------------------


ToolImpl = Callable[..., Dict[str, Any]]


@dataclass(frozen=True)
class ToolSpec:
	name: str
	description: str
	parameters: Dict[str, Any]
	impl: ToolImpl


def get_tools() -> List[ToolSpec]:
	return [
		ToolSpec(
			name="web_search",
			description="Search the web for recent information using DuckDuckGo.",
			parameters={
				"type": "object",
				"properties": {
					"query": {"type": "string", "description": "Search query in natural language"},
					"max_results": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5},
					"region": {"type": "string", "default": "wt-wt"},
				},
				"required": ["query"],
			},
			impl=tool_web_search,
		),
		ToolSpec(
			name="web_fetch",
			description="Fetch a webpage and extract readable text.",
			parameters={
				"type": "object",
				"properties": {
					"url": {"type": "string", "description": "Absolute URL to fetch"},
					"timeout_seconds": {"type": "integer", "minimum": 5, "maximum": 60, "default": 20},
					"max_chars": {"type": "integer", "minimum": 500, "maximum": 8000, "default": 4000},
				},
				"required": ["url"],
			},
			impl=tool_web_fetch,
		),
	]


def get_tools_schema() -> List[Dict[str, Any]]:
	return [
		{
			"type": "function",
			"function": {"name": spec.name, "description": spec.description, "parameters": spec.parameters},
		}
		for spec in get_tools()
	]


def run_tool(name: str, arguments: Dict[str, Any]) -> Tuple[bool, str]:
	"""Execute a tool by name and return (ok, string_result) for tool message content."""
	tool_by_name: Dict[str, ToolSpec] = {spec.name: spec for spec in get_tools()}
	spec = tool_by_name.get(name)
	if not spec:
		return False, json.dumps({"ok": False, "error": f"Unknown tool: {name}"})
	try:
		result = spec.impl(**arguments)
		return True, json.dumps(result, ensure_ascii=False)
	except Exception as exc:
		return False, json.dumps({"ok": False, "error": str(exc)})

