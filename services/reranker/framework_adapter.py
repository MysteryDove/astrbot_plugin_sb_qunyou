"""
Framework reranker adapter.

Wraps AstrBot's ``RerankProvider`` behind the plugin's ``IRerankProvider``
interface with query truncation and error translation.
"""
from __future__ import annotations

from typing import Any, List, Optional

from astrbot.api import logger

from .base import IRerankProvider, RerankResult, RerankProviderError

# Most rerank APIs enforce query length limits (256-512 tokens).
_MAX_QUERY_CHARS = 512


class FrameworkRerankAdapter(IRerankProvider):
    """Adapter: AstrBot RerankProvider → IRerankProvider."""

    def __init__(self, provider: Any) -> None:
        if provider is None:
            raise ValueError("provider must not be None")
        self._provider = provider

    async def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
    ) -> List[RerankResult]:
        try:
            # Truncate query to prevent API errors
            if len(query) > _MAX_QUERY_CHARS:
                query = query[:_MAX_QUERY_CHARS]

            results = await self._provider.rerank(query, documents, top_n)
            return [
                RerankResult(
                    index=r.index,
                    relevance_score=r.relevance_score,
                )
                for r in results
            ]
        except Exception as e:
            raise RerankProviderError(
                f"Framework rerank failed: {e}"
            ) from e

    def get_model_name(self) -> str:
        try:
            return self._provider.get_model()
        except Exception:
            return "<unknown>"

    async def close(self) -> None:
        pass  # Framework manages lifecycle
