"""Ecommerce Specialized Agent implementation."""
import json
import logging
from typing import AsyncGenerator, Optional
from app.agents.base import BaseAgent
from app.schemas.payload import ChatRequest, ChatResponse

logger = logging.getLogger("ai_gateway.agent.ecommerce")


class EcommerceAgent(BaseAgent):
    """Specialized AI Agent for product catalog inquiries, stock availability,

    pricing, product recommendations, and e-commerce guidance.
    """

    def __init__(self, agent_id: str = "ecommerce") -> None:
        super().__init__(
            agent_id=agent_id,
            name="E-Commerce & Catalog Agent",
            description="Responde sobre catálogo de productos, precios, disponibilidad y recomendaciones comerciales.",
            capabilities=[
                "product_search",
                "price_inquiry",
                "stock_check",
                "product_recommendation",
                "purchase_guidance",
            ],
        )

    async def get_system_instruction(self, request: ChatRequest) -> str:
        """Returns specialized persona and constraints for the E-Commerce Agent."""
        return (
            "Eres el Asistente Experto de E-Commerce y Catálogo de Productos. "
            "Tu misión es ayudar a los clientes a encontrar productos, resolver dudas sobre especificaciones, "
            "precios, disponibilidad en stock y guiarles en su decisión de compra.\n\n"
            "Pautas de respuesta:\n"
            "1. Utiliza ÚNICAMENTE la información provista en la sección [Context / Grounding Data] para detalles de precios, monedas e inventario.\n"
            "2. Sé servicial, dinámico y presenta los productos con viñetas claras, destacando el nombre, precio y características principales.\n"
            "3. Si un producto no se encuentra en el catálogo provisto, indícalo cortésmente y ofrece sugerir alternativas similares disponibles.\n"
            "4. Responde en el idioma del usuario (español por defecto)."
        )

    async def get_context_augmentation(self, request: ChatRequest) -> Optional[str]:
        """Queries the catalog via Django API based on the user's query keywords."""
        try:
            products = await self.django_service.search_catalog(query=request.message, limit=5)
            if not products:
                return "No products matched the exact query in the catalog."
            return f"Matching Catalog Products:\n{json.dumps(products, ensure_ascii=False, indent=2)}"
        except Exception as exc:
            logger.warning("Error fetching e-commerce catalog context: %s", exc)
            return None

    async def process(self, request: ChatRequest) -> ChatResponse:
        """Processes a chat request and returns a complete response."""
        return await self._execute_process(request)

    async def process_stream(self, request: ChatRequest) -> AsyncGenerator[str, None]:
        """Processes a chat request and yields SSE chunk tokens."""
        async for token in self._execute_process_stream(request):
            yield token
