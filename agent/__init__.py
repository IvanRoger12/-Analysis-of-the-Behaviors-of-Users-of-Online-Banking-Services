"""AI Agent package providing a tool-using conversational assistant.

Modules:
- config: Environment configuration loading and defaults
- llm_client: OpenAI-compatible LLM client wrapper
- tools: Pluggable tool implementations and schemas
- memory: Simple persistent memory store (JSON file)
- agent: Core agent loop orchestrating LLM and tool calls
- cli: Typer-based CLI entry point
"""

__all__ = [
	"config",
	"llm_client",
	"tools",
	"memory",
	"agent",
	"cli",
]
