from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv


@dataclass
class AgentConfig:
	provider: str = "echo"
	model: Optional[str] = None
	openai_api_key: Optional[str] = None
	ollama_host: Optional[str] = None


def load_config() -> AgentConfig:
	load_dotenv()
	provider = os.getenv("AGENT_PROVIDER", "echo").strip().lower()
	model = os.getenv("AGENT_MODEL")
	openai_api_key = os.getenv("OPENAI_API_KEY")
	ollama_host = os.getenv("OLLAMA_HOST")
	return AgentConfig(
		provider=provider,
		model=model,
		openai_api_key=openai_api_key,
		ollama_host=ollama_host,
	)
