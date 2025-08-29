from __future__ import annotations

from typing import Any, Dict, List, Optional

from .config import AgentConfig
from .llm_client import LLMClient
from .memory import MemoryStore
from .tools import get_tools_schema, run_tool


class Agent:
	"""Core agent loop coordinating the LLM, memory, and tools."""

	def __init__(self, config: AgentConfig, memory: MemoryStore) -> None:
		self._config = config
		self._memory = memory
		self._llm = LLMClient(config)

	def _build_system_message(self, system_prompt: Optional[str]) -> Dict[str, Any]:
		prompt = system_prompt or (
			"You are a helpful AI agent. Use tools when helpful."
		)
		return {"role": "system", "content": prompt}

	def run_once(self, user_input: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
		"""Run the agent for a single user input, potentially using tools for several steps.

		Returns the final assistant message dict.
		"""
		messages: List[Dict[str, Any]] = []
		persisted = self._memory.messages
		if not persisted or persisted[0].get("role") != "system":
			messages.append(self._build_system_message(system_prompt))
		else:
			messages.append(persisted[0])
		# add prior conversation excluding any prior system we already added
		for m in persisted[1:] if messages and messages[0] == persisted[0] else persisted:
			messages.append(m)
		messages.append({"role": "user", "content": user_input})

		tools_schema = get_tools_schema() if self._config.use_tools else None
		steps = 0
		assistant_message: Optional[Dict[str, Any]] = None
		while steps < self._config.max_steps:
			steps += 1
			assistant_message = self._llm.chat(messages=messages, tools_schema=tools_schema)
			messages.append(assistant_message)
			tool_calls = assistant_message.get("tool_calls") if isinstance(assistant_message, dict) else None
			if not tool_calls:
				break
			# Handle tool calls one by one
			for tool_call in tool_calls:
				tool_name = tool_call.get("function", {}).get("name")
				arg_json = tool_call.get("function", {}).get("arguments", "{}")
				args = self._llm.parse_tool_arguments(arg_json)
				ok, content = run_tool(tool_name, args)
				messages.append(
					{
						"role": "tool",
						"tool_call_id": tool_call.get("id", ""),
						"name": tool_name,
						"content": content,
					}
				)

		# Persist messages back to memory
		self._memory.clear()
		for m in messages:
			self._memory.append(m)

		return assistant_message or {"role": "assistant", "content": ""}

