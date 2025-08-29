from __future__ import annotations

from typing import Iterable, Optional

import ollama

from .base import LLMProvider


class OllamaProvider(LLMProvider):
	def __init__(self, host: Optional[str], model: Optional[str]) -> None:
		super().__init__(model=model)
		if host:
			ollama.set_host(host)

	def generate(self, prompt: str) -> str:
		model_name = self.model or "llama3.1"
		response = ollama.chat(model=model_name, messages=[{"role": "user", "content": prompt}])
		return response.get("message", {}).get("content", "")

	def stream(self, prompt: str) -> Iterable[str]:
		model_name = self.model or "llama3.1"
		for event in ollama.chat(
			model=model_name,
			messages=[{"role": "user", "content": prompt}],
			stream=True,
		):
			message = event.get("message", {})
			content = message.get("content")
			if content:
				yield content
