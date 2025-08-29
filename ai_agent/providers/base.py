from __future__ import annotations

from typing import Iterable, Optional


class LLMProvider:
	def __init__(self, model: Optional[str] = None) -> None:
		self.model = model

	def generate(self, prompt: str) -> str:
		raise NotImplementedError

	def stream(self, prompt: str) -> Iterable[str]:
		raise NotImplementedError
