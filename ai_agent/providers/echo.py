from __future__ import annotations

from typing import Iterable

from .base import LLMProvider


class EchoProvider(LLMProvider):
	def generate(self, prompt: str) -> str:
		return f"[echo] {prompt}"

	def stream(self, prompt: str) -> Iterable[str]:
		yield from (token for token in f"[echo] {prompt}")
