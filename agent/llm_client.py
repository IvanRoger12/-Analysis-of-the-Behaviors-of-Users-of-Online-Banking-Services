from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from openai import OpenAI

from .config import AgentConfig


class LLMClient:
	"""Thin wrapper around OpenAI's Chat Completions API with tool calling support."""

	def __init__(self, config: AgentConfig) -> None:
		self._config = config
		self._client = OpenAI()

	def chat(
		self,
		messages: List[Dict[str, Any]],
		tools_schema: Optional[List[Dict[str, Any]]] = None,
		tool_choice: str | Dict[str, Any] = "auto",
	) -> Dict[str, Any]:
		"""Send a chat completion request.

		Returns the first choice's message dict, including potential tool_calls.
		"""
		response = self._client.chat.completions.create(
			model=self._config.model,
			messages=messages,
			tools=tools_schema or None,
			tool_choice=tool_choice,
			temperature=0.2,
		)
		message = response.choices[0].message
		# Convert pydantic message to plain dict (ensure JSON-serializable)
		message_dict = {
			"role": message.role,
			"content": message.content or "",
		}
		if getattr(message, "tool_calls", None):
			tool_calls: List[Any] = message.tool_calls
			message_dict["tool_calls"] = [
				{
					"id": tc.id,
					"type": tc.type,
					"function": {
						"name": tc.function.name,
						"arguments": tc.function.arguments,
					},
				}
				for tc in tool_calls
			]
		return message_dict

	@staticmethod
	def parse_tool_arguments(arguments_json: str) -> Dict[str, Any]:
		"""Parse the function call arguments safely."""
		try:
			return json.loads(arguments_json) if arguments_json else {}
		except Exception:
			# Fallback: return empty args on malformed JSON
			return {}
