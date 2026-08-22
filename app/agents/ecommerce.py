"""Ecommerce Specialized Agent implementation with smart product extraction, Markdown knowledge grounding, and live catalog search."""
import json
import logging
import re
from typing import AsyncGenerator, Optional
from app.agents.base import BaseAgent
from app.schemas.payload import ChatRequest, ChatResponse
from app.services.django_api import DjangoAPIService
from app.services.knowledge_base import KnowledgeBaseService, get_knowledge_base_service
from app.services.llm_client import LLMClientService
from app.services.memory import RedisMemoryService

logger = logging.getLogger("ai_gateway.agent.ecommerce")

# Common conversational stopwords in Spanish and English to filter out from product search queries
STOPWORDS = {
    "hola", "buenas", "buenos", "dias", "días", "tardes", "noches", "saludos",
    "me", "interesa", "interesan", "quisiera", "gustaria", "gustaría", "deseo",
    "el", "la", "los", "las", "un", "una", "unos", "unas",
    "producto", "productos", "item", "items", "articulo", "artículo", "articulos", "artículos",
    "servicio", "servicios", "curso", "cursos",
    "precio", "precios", "costo", "costos", "valor", "valores", "cuanto", "cuánto", "cuesta", "cuestan",
    "tienen", "tiene", "hay", "disponible", "disponibles", "disponibilidad", "stock",
    "detalles", "detalle", "info", "información", "informacion", "características", "caracteristicas",
    "sobre", "de", "del", "en", "para", "por", "favor", "que", "qué", "y", "o", "a", "al", "con",
    "ver", "consultar", "comprar", "adquirir", "contratar", "saber", "mas", "más",
}


class EcommerceAgent(BaseAgent):
    """Specialized AI Agent for product catalog inquiries, stock availability,
    pricing, business policies (returns, warranties, shipping, payments), and e-commerce guidance.
    """

    def __init__(
        self,
        agent_id: str = "ecommerce",
        llm_service: Optional[LLMClientService] = None,
        memory_service: Optional[RedisMemoryService] = None,
        django_service: Optional[DjangoAPIService] = None,
        knowledge_base_service: Optional[KnowledgeBaseService] = None,
    ) -> None:
        super().__init__(
            agent_id=agent_id,
            name="E-Commerce & Catalog Agent",
            description="Responde sobre catálogo de productos, precios, disponibilidad, políticas de compra, envíos y reembolsos.",
            capabilities=[
                "product_search",
                "price_inquiry",
                "stock_check",
                "product_recommendation",
                "purchase_guidance",
                "business_policies",
            ],
            llm_service=llm_service,
            memory_service=memory_service,
            django_service=django_service,
        )
        self.knowledge_base_service = knowledge_base_service or get_knowledge_base_service()

    def extract_search_terms(self, message: str) -> list[str]:
        """Extracts candidate product names and keywords from conversational user queries.

        Handles quotes, prefix patterns ('producto ...', 'curso ...'), price tags ('($49.99)'),
        and filters conversational noise.
        """
        candidates: list[str] = []

        # 1. Detect text inside quotation marks: "Producto", 'Producto', “Producto”, «Producto»
        quoted_matches = re.findall(r'["\'“«]([^"\'”»]+)["\'”»]', message)
        for match in quoted_matches:
            cleaned = match.strip()
            # Remove trailing price tags like " ($49.99)" or " - $49"
            cleaned = re.sub(r'[\(\[\-]?\s*\$[\d.,]+\s*[\)\]]?', '', cleaned).strip()
            if cleaned and len(cleaned) >= 2:
                candidates.append(cleaned)

        # Conversational prefixes to strip from beginning of captured pattern
        LEADING_STOPWORDS = {"de", "del", "el", "la", "los", "las", "un", "una", "en", "para", "sobre", "que", "y"}

        # 2. Detect common intent phrases: e.g., 'me interesa el producto X', 'curso de X'
        pattern_matches = re.findall(
            r'(?:producto|curso|servicio|articulo|artículo|sobre|interesa|precio de|stock de)\s+([^?.!,;\n]+)',
            message,
            re.IGNORECASE,
        )
        for match in pattern_matches:
            cleaned = re.sub(r'[\(\[\-]?\s*\$[\d.,]+\s*[\)\]]?', '', match).strip()
            # Filter leading conversational stopwords from start
            words = [w for w in re.split(r'\s+', cleaned) if w]
            while words and re.sub(r'^\W+|\W+$', '', words[0].lower()) in LEADING_STOPWORDS:
                words.pop(0)
            if words:
                candidate = " ".join(words).strip()
                if candidate and len(candidate) >= 2 and candidate not in candidates:
                    candidates.append(candidate)

        # 3. Clean tokenized keywords from entire message as fallback candidate
        all_words = re.findall(r'\b[a-zA-ZáéíóúÁÉÍÓÚñÑ0-9\-_]{3,}\b', message)
        meaningful_tokens = [w for w in all_words if w.lower() not in STOPWORDS]
        if meaningful_tokens:
            token_candidate = " ".join(meaningful_tokens)
            if token_candidate not in candidates:
                candidates.append(token_candidate)

        return candidates

    async def get_system_instruction(self, request: ChatRequest) -> str:
        """Returns specialized persona and constraints for the E-Commerce Agent."""
        return (
            "Eres el Asistente Experto de E-Commerce, Catálogo y Políticas Comerciales de la Tienda. "
            "Tu misión es ayudar a los clientes a encontrar productos, resolver dudas sobre especificaciones, "
            "precios, disponibilidad en stock, políticas de compra, envíos, métodos de pago, cupones y devoluciones.\n\n"
            "Pautas de respuesta:\n"
            "1. Utiliza la sección [Live Catalog / Database Grounding] para responder con precisión sobre nombres de productos, precios, monedas y stock en tiempo real.\n"
            "2. Utiliza ESTRICTAMENTE la sección [Business Context & Policies] para responder sobre políticas de la tienda (plazos de entrega, costos de envío, métodos de pago aceptados, cupones, cancelaciones, devoluciones y soporte). No inventes políticas ni asumas reglas que contradigan dicho contexto.\n"
            "3. Cuando el usuario pregunte por un producto y sus condiciones comerciales, combina ambas fuentes de forma armónica, clara y estructurada.\n"
            "4. Sé servicial, dinámico y presenta los productos con viñetas claras destacando nombre, precio y disponibilidad.\n"
            "5. Responde siempre en el idioma del usuario (español por defecto)."
        )

    async def get_context_augmentation(self, request: ChatRequest) -> Optional[str]:
        """Loads Markdown business context and queries live Django database catalog."""
        context_blocks: list[str] = []

        # 1. Load Business Context & Policies from Markdown knowledge base
        try:
            business_context = await self.knowledge_base_service.get_ecommerce_context()
            if business_context:
                context_blocks.append(f"[Business Context & Policies]:\n{business_context}")
        except Exception as exc:
            logger.warning("Error loading business context from knowledge base: %s", exc)

        # 2. Query Live Catalog from Django database/API
        try:
            search_candidates = self.extract_search_terms(request.message)
            products: list[dict] = []

            # Stage 1: Try searching with extracted candidates (quoted names, specific patterns)
            for candidate in search_candidates:
                results = await self.django_service.search_catalog(query=candidate, limit=5)
                if results:
                    products = results
                    logger.info("Catalog match found using candidate '%s': %d products", candidate, len(products))
                    break

            # Stage 2: If no candidate yielded results, try raw message
            if not products:
                products = await self.django_service.search_catalog(query=request.message, limit=5)

            # Stage 3: If still empty, fetch representative/featured catalog for complete grounding
            if not products:
                logger.info("No direct catalog match; fetching representative catalog for grounding.")
                products = await self.django_service.search_catalog(query=None, limit=6)

            if products:
                catalog_json = json.dumps(products, ensure_ascii=False, indent=2)
                context_blocks.append(f"[Live Catalog / Database Grounding]:\n{catalog_json}")
            else:
                context_blocks.append("[Live Catalog / Database Grounding]:\nNo products currently available in catalog.")

        except Exception as exc:
            logger.warning("Error fetching e-commerce catalog context: %s", exc)

        if not context_blocks:
            return None

        return "\n\n".join(context_blocks)

    async def process(self, request: ChatRequest) -> ChatResponse:
        """Processes a chat request and returns a complete response."""
        return await self._execute_process(request)

    async def process_stream(self, request: ChatRequest) -> AsyncGenerator[str, None]:
        """Processes a chat request and yields SSE chunk tokens."""
        async for token in self._execute_process_stream(request):
            yield token
