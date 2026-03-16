"""
Reranker provider interface and value objects.

Defines the abstract contract for document reranking, aligned with
AstrBot framework's ``RerankProvider`` interface.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class RerankResult:
    """Single reranking result.

    Attributes:
        index: Original index in the candidate document list.
        relevance_score: Relevance score assigned by the reranker.
    """
    index: int
    relevance_score: float


class IRerankProvider(ABC):
    """Abstract reranker provider interface."""

    @abstractmethod
    async def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
    ) -> List[RerankResult]:
        """Rerank documents by relevance to the query.

        Returns sorted list of RerankResult (highest relevance first).
        """

    @abstractmethod
    def get_model_name(self) -> str:
        """Return the model identifier string."""

    async def close(self) -> None:
        """Release resources (no-op by default)."""


class RerankProviderError(Exception):
    """Raised when a reranker provider encounters an error."""
