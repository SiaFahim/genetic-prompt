import hashlib
import os
from typing import Optional


class DiskCache:
    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def _path(self, key: str) -> str:
        return os.path.join(self.cache_dir, f"{key}.txt")

    def __contains__(self, key: str) -> bool:
        return os.path.exists(self._path(key))

    def __getitem__(self, key: str) -> Optional[str]:
        p = self._path(key)
        if not os.path.exists(p):
            raise KeyError(key)
        with open(p, "r") as f:
            return f.read()

    def __setitem__(self, key: str, value: str) -> None:
        p = self._path(key)
        with open(p, "w") as f:
            f.write(value)

    @staticmethod
    def key_for(prompt_text: str) -> str:
        return hashlib.md5(prompt_text.encode()).hexdigest()

