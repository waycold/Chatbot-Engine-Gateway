"""Knowledge Base service for loading, caching and managing Markdown business context."""
import asyncio
import logging
from pathlib import Path
import time
from typing import Any, Optional
from app.core.config import settings

logger = logging.getLogger("ai_gateway.knowledge_base")

DEFAULT_ECOMMERCE_FALLBACK_CONTEXT = """# Información de Negocio y Políticas (Fallback)

- **Nombre:** AI Solutions & E-Commerce Store
- **Productos:** Servicios Cloud, Cursos de FastAPI y Microservicios, Módulos LLM y Consultoría de Arquitectura.
- **Envíos:** Productos digitales con entrega inmediata por email (máx 5 min). Consultorías agendadas vía Calendly.
- **Garantías:** 14 días corridos de garantía de satisfacción para productos digitales con reembolso del 100%.
- **Métodos de Pago:** Tarjetas de crédito/débito (Visa, Mastercard, AMEX), Transferencia bancaria (10% de descuento) y Criptomonedas (USDT, BTC).
- **Atención:** Lunes a Viernes de 09:00 a 18:00 (UTC-3). Contacto: soporte@techcommerce.example.com.
"""


class KnowledgeBaseService:
    """Asynchronous service for loading, caching, and managing Markdown-based business context.

    Features timestamp-aware in-memory caching to avoid redundant disk I/O while automatically
    reflecting edits made to the .md files without restarting the server.
    """

    def __init__(self, default_ecommerce_path: Optional[str] = None) -> None:
        self.ecommerce_path = default_ecommerce_path or settings.ECOMMERCE_CONTEXT_PATH
        # Cache structure: {file_path_str: {"content": str, "mtime": float, "last_checked": float}}
        self._cache: dict[str, dict[str, Any]] = {}
        self._cache_ttl_seconds: float = 10.0  # Check mtime at most every 10s

    def _sync_read_file(self, path: Path) -> tuple[Optional[str], float]:
        """Synchronous helper executed in threadpool to read file and mtime."""
        if not path.is_file():
            return None, 0.0
        try:
            mtime = path.stat().st_mtime
            content = path.read_text(encoding="utf-8").strip()
            return content, mtime
        except Exception as exc:
            logger.warning("Error reading knowledge file at %s: %s", path, exc)
            return None, 0.0

    async def load_markdown_file(self, file_path: Optional[str] = None) -> str:
        """Asynchronously loads a Markdown file with mtime-aware in-memory caching.

        Args:
            file_path: Relative or absolute path to the Markdown file.

        Returns:
            The file content as string, or fallback context if file is missing/empty.
        """
        target_path_str = file_path or self.ecommerce_path
        path = Path(target_path_str)

        now = time.time()
        cached = self._cache.get(target_path_str)

        # Check if cache is still fresh within TTL window
        if cached and (now - cached.get("last_checked", 0.0)) < self._cache_ttl_seconds:
            return str(cached["content"])

        # Run non-blocking file read in threadpool
        content, mtime = await asyncio.to_thread(self._sync_read_file, path)

        if content:
            self._cache[target_path_str] = {
                "content": content,
                "mtime": mtime,
                "last_checked": now,
            }
            logger.debug("Loaded and cached knowledge base from %s (size: %d bytes)", target_path_str, len(content))
            return content

        # If cached version exists but disk read failed momentarily, return cached
        if cached:
            return str(cached["content"])

        logger.warning(
            "Knowledge base file not found or empty at '%s'. Using built-in fallback.", target_path_str
        )
        return DEFAULT_ECOMMERCE_FALLBACK_CONTEXT

    async def get_ecommerce_context(self) -> str:
        """Retrieves the business context Markdown for the EcommerceAgent."""
        return await self.load_markdown_file(self.ecommerce_path)

    def clear_cache(self) -> None:
        """Clears all cached knowledge base files."""
        self._cache.clear()


# Global singleton instance
_knowledge_base_service: Optional[KnowledgeBaseService] = None


def get_knowledge_base_service() -> KnowledgeBaseService:
    """Returns the singleton instance of KnowledgeBaseService."""
    global _knowledge_base_service
    if _knowledge_base_service is None:
        _knowledge_base_service = KnowledgeBaseService()
    return _knowledge_base_service
