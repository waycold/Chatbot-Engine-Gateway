"""Ecommerce Specialized Agent implementation with smart product extraction and robust catalog grounding."""
import json
import logging
import re
from typing import AsyncGenerator, Optional
from app.agents.base import BaseAgent
from app.schemas.payload import ChatRequest, ChatResponse

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
            "Eres el Asistente Experto de E-Commerce y Catálogo de Productos. "
            "Tu misión es ayudar a los clientes a encontrar productos, resolver dudas sobre especificaciones, "
            "precios, disponibilidad en stock y guiarles en su decisión de compra.\n\n"
            "Pautas de respuesta:\n"
            "1. Utiliza la información provista en la sección [Context / Grounding Data] para responder con precisión sobre nombres, precios, monedas y disponibilidad.\n"
            "2. Sé servicial, dinámico y presenta los productos de forma estructurada con viñetas claras, destacando el nombre, precio y características principales.\n"
            "3. Si el usuario pregunta por un producto específico, proporciona su detalle completo. Si no coincide exactamente, presenta las opciones más afines del catálogo.\n"
            "4. Responde siempre en el idioma del usuario (español por defecto)."
        )

    async def get_context_augmentation(self, request: ChatRequest) -> Optional[str]:
        """Queries catalog via Django API with smart product extraction and multi-stage fallback."""
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

            if not products:
                return "No products currently available in the catalog."

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
