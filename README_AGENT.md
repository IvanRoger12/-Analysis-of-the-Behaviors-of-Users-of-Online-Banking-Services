AI Agent Quickstart

Setup
- Ensure Python 3.10+ is installed
- Create and activate a virtual environment, then install dependencies:
```
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```
- Configure environment (you can use a .env file):
```
export OPENAI_API_KEY=sk-...
export AGENT_MODEL=gpt-4o-mini
```

Usage
- Single prompt run:
```
python -m agent.cli chat --prompt "Bonjour, résume les news IA d'aujourd'hui" --max-steps 3
```
- Interactive chat session:
```
python -m agent.cli chat
```
- Show memory contents:
```
python -m agent.cli memory
```

Notes
- Tools available: web_search, web_fetch
- Memory stored at ./agent_data/memory.json (configurable via AGENT_MEMORY_PATH)
- Other env vars: AGENT_MAX_STEPS, AGENT_USE_TOOLS, AGENT_REQUEST_TIMEOUT
