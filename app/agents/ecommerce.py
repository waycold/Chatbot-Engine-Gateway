"""Ecommerce Specialized Agent implementation with hybrid Markdown business context, live catalog grounding, reviews, and semantic search."""
import json
import logging
import re
from typing import AsyncGenerator, Optional
from app.agents.base import BaseAgent
from app.schemas.payload import ChatRequest, ChatResponse
from app.services.knowledge_base import KnowledgeBaseService, get_knowledge_base_service

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
    """Specialized AI Agent for e-commerce, combining static/editable Markdown business

    knowledge (shipping, refunds, financing, FAQs) with live database catalog grounding (products, prices, stock, reviews).
    """

    def __init__(
        self,
        agent_id: str = "ecommerce",
        knowledge_service: Optional[KnowledgeBaseService] = None,
    ) -> None:
        super().__init__(
            agent_id=agent_id,
            name="E-Commerce & Business Agent",
            description="Responde sobre catálogo de productos, precios, disponibilidad en stock, políticas de envío, devoluciones, reseñas y métodos de pago.",
            capabilities=[
                "product_search",
                "semantic_search",
                "price_inquiry",
                "stock_check",
                "customer_reviews",
                "shipping_policies",
                "refund_policies",
                "payment_methods",
                "faq_resolution",
                "purchase_guidance",
            ],
        )
        self.knowledge_base_service = knowledge_service or get_knowledge_base_service()

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
        """Returns specialized persona and constraints for the E-Commerce & Business Agent."""
        return (
            "Eres el Asistente Experto de E-Commerce y Consultas Comerciales de la tienda.\n"
            "Tu misión es brindar atención integral a los clientes respondiendo preguntas sobre:\n"
            "1. Información de la empresa, métodos de pago, financiación, políticas de envío, devoluciones y garantías (usando la sección de Políticas y Base de Conocimiento).\n"
            "2. Búsqueda de productos, características técnicas, precios exactos, monedas, disponibilidad en stock y opiniones de clientes (usando la sección de Catálogo / Base de Datos).\n\n"
            "Pautas de respuesta:\n"
            "- Sé servicial, cordial, dinámico y profesional.\n"
            "- Presenta opciones y productos de forma estructurada (viñetas, precios claros, enlaces o condiciones relevantes).\n"
            "- Para preguntas institucionales o de políticas (envíos, devoluciones, formas de pago), responde con base estricta en el documento de políticas provisto.\n"
            "- Para preguntas sobre productos específicos o reseñas, utiliza los datos en tiempo real del catálogo provisto.\n"
            "- Responde siempre en el mismo idioma del usuario (español por defecto)."
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

            # 3. If user inquires about reviews or ratings, query reviews summary
            msg_lower = request.message.lower()
            if any(k in msg_lower for k in ["reseña", "reseñas", "review", "reviews", "calificación", "calificacion", "opinión", "opiniones", "estrellas"]):
                reviews_data = await self.django_service.get_customer_reviews_summary(user_token=request.user_token)
                context_blocks.append(f"[Customer Reviews & Ratings Summary]:\n{json.dumps(reviews_data, ensure_ascii=False, indent=2)}")

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
