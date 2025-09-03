import json
import os
from dataclasses import dataclass
from typing import Any, Dict

try:
    from dotenv import load_dotenv  # optional; available via python-dotenv
except Exception:
    load_dotenv = None


@dataclass
class Config:
    raw: Dict[str, Any]

    @property
    def model_provider(self) -> str:
        return str(self.raw.get("model", {}).get("provider", "openai"))

    @property
    def random_seed(self) -> int:
        return int(self.raw.get("random_seed", 42))

    @property
    def model_name(self) -> str:
        return str(self.raw.get("model", {}).get("model_name", "gpt-4o"))

    @property
    def temperature(self) -> float:
        return float(self.raw.get("model", {}).get("temperature", 0))

    @property
    def max_tokens(self) -> int:
        return int(self.raw.get("model", {}).get("max_tokens", 200))

    @property
    def evaluation(self) -> Dict[str, Any]:
        return dict(self.raw.get("evaluation", {}))

    @property
    def paths(self) -> Dict[str, str]:
        return dict(self.raw.get("paths", {}))

    @property
    def api_keys(self) -> Dict[str, str]:
        api = dict(self.raw.get("api", {}))
        openai_var = api.get("openai_api_key_var", "OPENAI_API_KEY")
        anthropic_var = api.get("anthropic_api_key_var", "ANTHROPIC_API_KEY")
        return {
            "openai": os.getenv(openai_var, ""),
            "anthropic": os.getenv(anthropic_var, ""),
        }


def load_config(config_path: str = "configs/experiment_config.json", load_env: bool = True) -> Config:
    if load_env and load_dotenv is not None:
        load_dotenv()
    with open(config_path, "r") as f:
        raw = json.load(f)
    cfg = Config(raw=raw)
    # Dynamic logging (no hardcoded values)
    print(f"Loaded config for provider={cfg.model_provider}, model={cfg.model_name}, temp={cfg.temperature}, max_tokens={cfg.max_tokens}")
    return cfg

