from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List


Message = Dict[str, Any]


@dataclass
class MemoryStore:
	"""Very simple JSON-based memory store for chat histories."""

	path: Path
	max_messages: int = 100
	_messages: List[Message] = field(default_factory=list)

	def load(self) -> None:
		if not self.path.exists():
			self._messages = []
			return
		try:
			self._messages = json.loads(self.path.read_text(encoding="utf-8"))
		except Exception:
			self._messages = []

	def save(self) -> None:
		data = self._messages[-self.max_messages :]
		self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

	@property
	def messages(self) -> List[Message]:
		return list(self._messages)

	def append(self, message: Message) -> None:
		self._messages.append(message)
		self.save()

	def clear(self) -> None:
		self._messages = []
		self.save()
