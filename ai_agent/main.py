from __future__ import annotations

import argparse
import sys
from typing import Iterable

from .config import load_config
from .providers.base import LLMProvider
from .providers.echo import EchoProvider
from .providers.openai_provider import OpenAIProvider
from .providers.ollama_provider import OllamaProvider


def build_provider(name: str, model: str | None) -> LLMProvider:
	config = load_config()
	provider = name.lower().strip()
	if provider == "openai":
		return OpenAIProvider(api_key=config.openai_api_key, model=model or config.model)
	if provider == "ollama":
		return OllamaProvider(host=config.ollama_host, model=model or config.model)
	return EchoProvider(model=model or config.model)


def print_stream(chunks: Iterable[str]) -> None:
	for chunk in chunks:
		print(chunk, end="", flush=True)
	print()


def repl(provider: LLMProvider) -> None:
	print("Type 'exit' or Ctrl+C to quit.")
	while True:
		try:
			user_input = input("You: ").strip()
			if user_input.lower() in {"exit", "quit"}:
				break
			print("Assistant:", end=" ")
			print_stream(provider.stream(user_input))
		except KeyboardInterrupt:
			print("\nBye!")
			break
		except Exception as exc:  # noqa: BLE001
			print(f"\n[error] {exc}")


def main(argv: list[str] | None = None) -> int:
	config = load_config()
	parser = argparse.ArgumentParser(description="Minimal AI Agent CLI")
	parser.add_argument("--provider", default=config.provider, help="echo | openai | ollama")
	parser.add_argument("--model", default=config.model, help="Model name for provider")
	args = parser.parse_args(argv)

	prov = build_provider(args.provider, args.model)
	repl(prov)
	return 0


if __name__ == "__main__":
	sys.exit(main())
