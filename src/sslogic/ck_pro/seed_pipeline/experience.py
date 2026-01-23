import json
from pathlib import Path
from typing import Iterable, List

EXPERIENCE_FILE = Path(__file__).resolve().parent / "experience.json"


class ExperienceManager:
    """Simple persistence layer for shared skill/experience notes."""

    def __init__(self, path: Path = EXPERIENCE_FILE):
        self.path = Path(path)
        self._cached: List[str] = []
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    self._cached = [str(item) for item in data if str(item).strip()]
                else:
                    self._cached = []
            except Exception:
                self._cached = []
        else:
            self._cached = []

    def all(self) -> List[str]:
        return list(self._cached)

    def add(self, notes: Iterable[str]) -> List[str]:
        added: List[str] = []
        for note in notes:
            if not note:
                continue
            text = str(note).strip()
            if not text or text in self._cached:
                continue
            self._cached.append(text)
            added.append(text)
        if added:
            self._persist()
        return added

    def _persist(self) -> None:
        self.path.write_text(
            json.dumps(self._cached, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


