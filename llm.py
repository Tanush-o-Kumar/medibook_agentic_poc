"""Thin client for a free local LLM served by Ollama (http://localhost:11434).

If Ollama is not running, every call returns None and the agents fall back
to deterministic rules - so the demo never breaks on stage.
"""
import json
import requests


class LocalLLM:
    def __init__(self, mode="auto", model="llama3.2", host="http://localhost:11434", timeout=90):
        self.mode, self.model, self.host, self.timeout = mode, model, host.rstrip("/"), timeout
        self._ok = None

    # ------------------------------------------------------------------ status
    def list_models(self):
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=2)
            r.raise_for_status()
            return [m["name"] for m in r.json().get("models", [])]
        except Exception:
            return None

    @property
    def enabled(self):
        if self.mode == "mock":
            return False
        if self._ok is None:
            models = self.list_models()
            self._ok = bool(models) and any(m.split(":")[0] == self.model.split(":")[0] for m in models)
        return self._ok

    @property
    def label(self):
        return f"Ollama · {self.model}" if self.enabled else "Rule-based mock"

    # ------------------------------------------------------------------ calls
    def _chat(self, system, user, as_json):
        payload = {"model": self.model, "stream": False, "options": {"temperature": 0},
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if as_json:
            payload["format"] = "json"
        r = requests.post(f"{self.host}/api/chat", json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()["message"]["content"]

    def json(self, system, user):
        if not self.enabled:
            return None
        try:
            out = json.loads(self._chat(system, user, as_json=True))
            return out if isinstance(out, dict) else None
        except Exception:
            return None

    def text(self, system, user):
        if not self.enabled:
            return None
        try:
            return self._chat(system, user, as_json=False).strip()
        except Exception:
            return None
