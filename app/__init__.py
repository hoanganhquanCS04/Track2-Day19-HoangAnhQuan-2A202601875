"""Lab 19 application package.

Loads `.env` on import. Without this, nothing in the repo ever read the file:
`app/embeddings.py` calls `os.getenv("EMBEDDING_BACKEND")`, `app/search.py`
calls `os.getenv("QDRANT_MODE")`, and the README documents both as the way to
switch stacks -- but no module ever called `load_dotenv()`, so editing `.env`
had no effect and every path silently kept the defaults. Setting the variable
worked only if you exported it in the shell first, which the README never says.

Real environment variables still win: `load_dotenv` does not override an
already-set variable, so `EMBEDDING_BACKEND=multilingual make benchmark` beats
whatever `.env` says -- which is what you want for a one-off experiment.
"""
from __future__ import annotations

from pathlib import Path

_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover — .env support is a convenience, not a requirement
    pass
else:
    if _ENV_FILE.exists():
        load_dotenv(_ENV_FILE, override=False)
