import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class AgentConfig:
	"""Configuration values for the AI agent.

	Values are read from environment variables with sensible defaults.
	"""
	model: str = os.getenv("AGENT_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
	openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
	max_steps: int = int(os.getenv("AGENT_MAX_STEPS", "5"))
	use_tools: bool = os.getenv("AGENT_USE_TOOLS", "true").lower() not in {"0", "false", "no"}
	request_timeout_seconds: int = int(os.getenv("AGENT_REQUEST_TIMEOUT", "60"))
	memory_path: Path = Path(os.getenv("AGENT_MEMORY_PATH", "./agent_data/memory.json")).expanduser()
	verbose: bool = os.getenv("AGENT_VERBOSE", "true").lower() not in {"0", "false", "no"}


def ensure_data_dir_exists(config: AgentConfig) -> None:
	"""Ensure the parent directory for memory storage exists."""
	config.memory_path.parent.mkdir(parents=True, exist_ok=True)


def load_config() -> AgentConfig:
	"""Load configuration from environment and ensure data directories exist."""
	config = AgentConfig()
	ensure_data_dir_exists(config)
	return config
