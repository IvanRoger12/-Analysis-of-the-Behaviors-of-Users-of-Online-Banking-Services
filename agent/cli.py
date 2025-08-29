from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from .agent import Agent
from .config import AgentConfig, load_config
from .memory import MemoryStore
from dotenv import load_dotenv


app = typer.Typer(add_completion=False, help="AI Agent CLI")
console = Console()
load_dotenv()


def _build_agent(
	model: Optional[str],
	max_steps: Optional[int],
	use_tools: Optional[bool],
	memory_path: Optional[Path],
	verbose: Optional[bool],
) -> Agent:
	base = load_config()
	if model:
		base.model = model
	if max_steps is not None:
		base.max_steps = max_steps
	if use_tools is not None:
		base.use_tools = use_tools
	if verbose is not None:
		base.verbose = verbose
	if memory_path is not None:
		base.memory_path = memory_path
		from .config import ensure_data_dir_exists
		ensure_data_dir_exists(base)
	mem = MemoryStore(path=base.memory_path, max_messages=200)
	mem.load()
	return Agent(config=base, memory=mem)


@app.command("chat")
def chat(
	prompt: Optional[str] = typer.Option(None, "--prompt", "-p", help="Single prompt to run once"),
	model: Optional[str] = typer.Option(None, help="Model name to use"),
	max_steps: int = typer.Option(None, help="Max tool-use steps per turn"),
	use_tools: Optional[bool] = typer.Option(None, help="Enable tool use (true/false)"),
	memory_path: Optional[Path] = typer.Option(None, help="Path to memory JSON file"),
	verbose: Optional[bool] = typer.Option(None, help="Verbose logs"),
	system_prompt: Optional[str] = typer.Option(None, help="Override system prompt for this session"),
	clear: bool = typer.Option(False, help="Clear conversation memory before starting"),
) -> None:
	"""Start an interactive chat session or run a single prompt."""
	agent = _build_agent(model, max_steps, use_tools, memory_path, verbose)
	if clear:
		agent._memory.clear()
		console.print("Memory cleared.")

	api_key = AgentConfig().openai_api_key
	if not api_key:
		console.print(Panel.fit("OPENAI_API_KEY is not set. Set it to enable LLM responses.", title="Missing API Key", style="red"))

	if prompt:
		resp = agent.run_once(prompt, system_prompt)
		console.print(Panel(resp.get("content", ""), title="Assistant"))
		raise typer.Exit(0)

	console.print(Panel("Type 'exit' or 'quit' to leave.", title="Interactive Chat"))
	while True:
		user = Prompt.ask("You")
		if user.strip().lower() in {"exit", "quit"}:
			break
		resp = agent.run_once(user, system_prompt)
		console.print(Panel(resp.get("content", ""), title="Assistant"))


@app.command("memory")
def memory_show(
	memory_path: Optional[Path] = typer.Option(None, help="Path to memory JSON file"),
) -> None:
	"""Print current memory JSON contents."""
	agent = _build_agent(None, None, None, memory_path, None)
	data = agent._memory.messages
	console.print_json(json.dumps(data, ensure_ascii=False))


def run():
	app()


if __name__ == "__main__":
	run()

