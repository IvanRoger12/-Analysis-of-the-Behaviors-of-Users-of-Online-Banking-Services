from __future__ import annotations

from typing import Iterable, Optional

from openai import OpenAI

from .base import LLMProvider


class OpenAIProvider(LLMProvider):
	def __init__(self, api_key: Optional[str], model: Optional[str]) -> None:
		super().__init__(model=model)
		self.client = OpenAI(api_key=api_key) if api_key else OpenAI()

	def generate(self, prompt: str) -> str:
		model_name = self.model or "gpt-4o-mini"
		response = self.client.chat.completions.create(
			model=model_name,
			messages=[{"role": "user", "content": prompt}],
		)
		return response.choices[0].message.content or ""

	def stream(self, prompt: str) -> Iterable[str]:
		model_name = self.model or "gpt-4o-mini"
		with self.client.chat.completions.create(
			model=model_name,
			messages=[{"role": "user", "content": prompt}],
			stream=True,
		) as stream:
			for event in stream:
				if event.choices and event.choices[0].delta and event.choices[0].delta.content:
					yield event.choices[0].delta.content
