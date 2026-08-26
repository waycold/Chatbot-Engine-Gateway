"""Ecommerce Specialized Agent implementation with hybrid Markdown business context, live catalog grounding, reviews, and semantic search."""
import json
import logging
import re
from typing import Any, AsyncGenerator, Optional
from app.agents.base import BaseAgent, EventSink
from app.agents.tools import CATALOG_RAG_TOOL_DECLARATIONS
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

    def get_tool_declarations(self, request: ChatRequest) -> list[dict[str, Any]]:
        """Returns the catalog RAG tool schemas — and nothing else.

        The analytics tools and, above all, the raw SQL console (`execute_raw_sql_sandbox`)
        are deliberately excluded and MUST stay excluded. This agent is the public,
        unauthenticated chat surface, and the RAG pipeline will soon ingest user-generated
        review text: a review body is an injection vector that reaches the model without
        the attacker ever joining the conversation. If the SQL console were reachable from
        here, a single poisoned review could exfiltrate the whole database. Making the
        schema structurally unreachable — rather than merely discouraged by a prompt — is
        the only defense that survives that threat model.

        Args:
            request: The incoming chat request.

        Returns:
            A fresh copy of the four catalog retrieval tool declarations.
        """
        # Return a copy, never the module-level list itself. Handing out the shared object
        # from a security-critical function means any caller that appends to the returned
        # list silently mutates the global catalog tool set for every agent in the process --
        # an in-process privilege escalation path that would put the SQL console right back.
        return list(CATALOG_RAG_TOOL_DECLARATIONS)

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
            "- Responde siempre en el mismo idioma del usuario (español por defecto).\n\n"
            "REGLAS OBLIGATORIAS SOBRE HERRAMIENTAS (no son sugerencias):\n"
            "1. ANTIALUCINACIÓN DE STOCK Y PRECIO: DEBES llamar a `check_stock_and_price` ANTES de "
            "afirmar CUALQUIER disponibilidad o precio al usuario, incluso si `semantic_catalog_search` "
            "ya devolvió esos datos. La búsqueda semántica optimiza recall y NO es fuente de verdad de "
            "disponibilidad ni de precio: esos valores pueden haber cambiado entre la indexación y este "
            "instante. Nunca inventes ni supongas stock, precio ni moneda; si no verificaste, no lo afirmes.\n"
            "2. DIVULGACIÓN OBLIGATORIA DE DEGRADACIÓN: si CUALQUIER resultado de herramienta trae "
            "`status: \"degraded\"`, tu respuesta DEBE ABRIR diciendo en lenguaje llano que hubo un "
            "problema técnico al buscar en el catálogo y que los resultados pueden estar incompletos, "
            "ANTES de listar ningún producto. No basta con registrarlo internamente: el usuario debe "
            "leerlo primero, porque de lo contrario tomará una decisión de compra creyendo que vio "
            "todo el catálogo cuando no fue así.\n"
            "3. FACETAS ANTES DE FILTRAR: llama a `list_catalog_facets` ANTES de filtrar "
            "`semantic_catalog_search` por `category` o `brand`, y usa EXACTAMENTE los valores que "
            "devuelva. Nunca inventes una categoría o marca que no exista en la base de datos."
        )

    async def get_context_augmentation(self, request: ChatRequest) -> Optional[str]:
        """Loads Markdown business context and queries live Django database catalog.

        Behaviour of the eager grounding blocks is unchanged. A machine-readable
        `[Catalog Retrieval Health]` block is appended when — and only when — a retrieval
        degradation was detected, so the model reliably sees it instead of having to infer
        it from a missing field.
        """
        context_blocks: list[str] = []
        degradation_reasons: list[str] = []

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
                if isinstance(reviews_data, dict) and reviews_data.get("status") == "degraded":
                    degradation_reasons.append(
                        str(reviews_data.get("degraded_reason") or "El resumen de reseñas llegó en modo degradado.")
                    )

        except Exception as exc:
            logger.warning("Error fetching e-commerce catalog context: %s", exc)
            degradation_reasons.append(
                f"La consulta al catálogo falló y el listado puede estar incompleto: {exc}"
            )

        # Machine-readable degradation flag: the system prompt requires the reply to OPEN
        # with a plain-language warning whenever this block reports degraded=true.
        if degradation_reasons:
            health_payload = {
                "degraded": True,
                "reasons": degradation_reasons,
                "instruction": (
                    "Los resultados de catálogo pueden estar incompletos. Debes advertirlo al usuario "
                    "en lenguaje llano al COMIENZO de tu respuesta, antes de listar productos."
                ),
            }
            context_blocks.append(
                f"[Catalog Retrieval Health]:\n{json.dumps(health_payload, ensure_ascii=False, indent=2)}"
            )

        if not context_blocks:
            return None

        return "\n\n".join(context_blocks)

    async def process(self, request: ChatRequest) -> ChatResponse:
        """Processes a chat request and returns a complete response."""
        return await self._execute_process(request)

    async def process_stream(
        self,
        request: ChatRequest,
        *,
        event_sink: Optional[EventSink] = None,
    ) -> AsyncGenerator[str, None]:
        """Processes a chat request and yields SSE chunk tokens.

        Args:
            request: The incoming chat request.
            event_sink: Optional tool-progress callback forwarded to the tool loop.
        """
        async for token in self._execute_process_stream(request, event_sink=event_sink):
            yield token
