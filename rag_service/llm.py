import json
from typing import Any, Dict, Optional
from urllib import error, request


class OllamaGenerator:
    def __init__(
        self,
        model: str = "llama3.1",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.1,
        timeout: int = 120,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.timeout = timeout

    def __call__(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        response = self._post_json("/api/generate", payload)
        text = response.get("response", "").strip()
        if not text:
            raise RuntimeError("Ollama returned an empty response")
        return text

    def _post_json(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        http_request = request.Request(
            self.base_url + path,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(http_request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
                if "error" in payload:
                    raise RuntimeError(str(payload["error"]))
                return payload
        except error.URLError as exc:
            raise RuntimeError(
                "Cannot reach Ollama. Start it with `ollama serve` and pull a model, "
                "for example `ollama pull llama3.1`."
            ) from exc


def make_generator(mode: str, ollama_model: str, ollama_url: str) -> Optional[OllamaGenerator]:
    if mode == "Ollama":
        return OllamaGenerator(model=ollama_model, base_url=ollama_url)
    return None
