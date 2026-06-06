import json
from pathlib import Path
from datetime import datetime


class ConversationMemory:
    def __init__(self, max_turns: int = 6, session_id: str = "default"):
        self.history    = []
        self.max_turns  = max_turns
        self.session_id = session_id
        self.created_at = datetime.now().isoformat()
        self.topics     = []  # track what was discussed

    def add(self, role: str, content: str):
        self.history.append({
            "role":       role,
            "content":    content,
            "timestamp":  datetime.now().isoformat()
        })
        # keep within max_turns
        max_msgs = self.max_turns * 2
        if len(self.history) > max_msgs:
            self.history = self.history[-max_msgs:]

    def get(self) -> list[dict]:
        # return format LLM expects (role + content only)
        return [{"role": m["role"], "content": m["content"]}
                for m in self.history]

    def add_topic(self, topic: str):
        if topic not in self.topics:
            self.topics.append(topic)

    def get_context_summary(self) -> str:
        """Return a short summary of what was discussed."""
        if not self.topics:
            return ""
        return f"Previously discussed: {', '.join(self.topics[-3:])}"

    def clear(self):
        self.history = []
        self.topics  = []

    def summary(self) -> str:
        turns = len(self.history) // 2
        if turns == 0:
            return "No conversation yet."
        return f"{turns} turn{'s' if turns != 1 else ''} in memory"

    def save(self, path: str = "indexes/session.json"):
        with open(path, "w") as f:
            json.dump({
                "session_id": self.session_id,
                "created_at": self.created_at,
                "history":    self.history,
                "topics":     self.topics
            }, f, indent=2)

    def load(self, path: str = "indexes/session.json"):
        if Path(path).exists():
            with open(path) as f:
                data = json.load(f)
            self.history    = data.get("history", [])
            self.topics     = data.get("topics", [])
            self.session_id = data.get("session_id", "default")