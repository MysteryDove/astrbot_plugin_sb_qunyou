"""
Reranker provider factory.

Creates ``IRerankProvider`` instances by resolving AstrBot framework
providers via ``context.get_provider_by_id(provider_id)``.
"""
from __future__ import annotations

from typing import Optional, TYPE_CHECKING

from astrbot.api import logger

from .base import IRerankProvider

if TYPE_CHECKING:
    from astrbot.api.star import Context


class RerankProviderFactory:
    """Factory for creating reranker provider instances."""

    @staticmethod
    def create(
        rerank_provider_id: Optional[str],
        context: "Context",
    ) -> Optional[IRerankProvider]:
        """Create a reranker provider from AstrBot context.

        Args:
            rerank_provider_id: The configured rerank provider ID.
            context: AstrBot plugin context.

        Returns:
            An IRerankProvider, or None if not configured.
        """
        if not rerank_provider_id:
            logger.debug("[RerankFactory] No rerank_provider_id, disabled")
            return None

        if context is None:
            logger.warning("[RerankFactory] Context is None, cannot resolve")
            return None

        try:
            provider = context.get_provider_by_id(rerank_provider_id)
        except Exception as e:
            logger.warning(f"[RerankFactory] Lookup failed: {e}")
            return None

        if provider is None:
            logger.warning(
                f"[RerankFactory] Provider '{rerank_provider_id}' not found"
            )
            return None

        # Validate it's a RerankProvider
        try:
            from astrbot.core.provider.provider import (
                RerankProvider as FrameworkRerankProvider,
            )
            if not isinstance(provider, FrameworkRerankProvider):
                logger.warning(
                    f"[RerankFactory] '{rerank_provider_id}' is "
                    f"{type(provider).__name__}, expected RerankProvider"
                )
                return None
        except ImportError:
            logger.debug("[RerankFactory] Cannot import FrameworkRerankProvider")

        from .framework_adapter import FrameworkRerankAdapter
        adapter = FrameworkRerankAdapter(provider)
        logger.info(
            f"[RerankFactory] Resolved: id={rerank_provider_id}, "
            f"model={adapter.get_model_name()}"
        )
        return adapter
