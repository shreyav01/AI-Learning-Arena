"""
utils/config.py
---------------
Centralizes all environment variable loading via python-dotenv.
All other modules import from here — no scattered os.getenv() calls.
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # OpenRouter LLM
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1/chat/completions"
    OPENROUTER_MODEL: str = os.getenv("OPENROUTER_MODEL", "openrouter/auto")

    # Embeddings (local sentence-transformers)
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # Chunking
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 500))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 50))

    # Retrieval
    TOP_K_CHUNKS: int = int(os.getenv("TOP_K_CHUNKS", 5))

    # Storage dirs
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "uploads")
    VECTOR_STORE_DIR: str = os.getenv("VECTOR_STORE_DIR", "vector_store")

    def validate(self):
        """Raise early if critical config is missing."""
        if not self.OPENROUTER_API_KEY:
            raise EnvironmentError(
                "OPENROUTER_API_KEY is not set. "
                "Copy .env.example → .env and add your key."
            )


settings = Settings()
