"""Tool Calling and Function Calling declarations for Google GenAI agents."""
import logging
from typing import Any, Callable, Coroutine, Iterable, Optional
from app.services.catalog_search import (
    find_similar_products_with_fallback,
    semantic_catalog_search_with_fallback,
)
from app.services.django_api import get_django_api_service

logger = logging.getLogger("ai_gateway.agents.tools")

# Name of the privileged read-only SQL console. Declared as a constant (instead of an
# inline literal at every call site) because two independent authorization layers must
# agree on exactly which tool is the privileged one.
SQL_SANDBOX_TOOL_NAME = "execute_raw_sql_sandbox"

# Schema definitions for Google GenAI function declarations
ANALYTICS_TOOL_DECLARATIONS = [
    {
        "name": "query_sales_analytics",
        "description": "Consulta métricas de ventas agregadas, ingresos totales, costos, ganancias brutas y márgenes % agrupados por dimensión.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "date_from": {
                    "type": "STRING",
                    "description": "Fecha de inicio en formato YYYY-MM-DD (opcional)",
                },
                "date_to": {
                    "type": "STRING",
                    "description": "Fecha de fin en formato YYYY-MM-DD (opcional)",
                },
                # TODO(hallazgo-D, diagnostico-plan-agentes-multi-agente.md): this enum has no
                # "product" option, so no combination of tool calls can literally answer a
                # "top N best-selling products" query (by units/revenue). Out of scope for this
                # round -- do NOT add "product" here without confirming the Django backend
                # endpoint actually supports a per-product breakdown. See the escalation ticket
                # in the Subagente 3 report for the two options under evaluation.
                "dimension": {
                    "type": "STRING",
                    "description": "Dimensión de agrupación: 'category', 'brand', 'supplier', 'payment_method', 'country', 'day', 'week', 'month', 'quarter'",
                    "enum": ["category", "brand", "supplier", "payment_method", "country", "day", "week", "month", "quarter"],
                },
            },
        },
    },
    {
        "name": "get_inventory_health",
        "description": "Consulta el estado de salud del inventario, stock crítico, productos agotados, valorización y velocidad de agotamiento (Runout Rate a 30 días).",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "status_filter": {
                    "type": "STRING",
                    "description": "Filtro de estado: 'all', 'critical', 'out_of_stock', 'healthy'",
                    "enum": ["all", "critical", "out_of_stock", "healthy"],
                },
                "limit": {
                    "type": "INTEGER",
                    "description": "Cantidad máxima de productos a listar (máximo 50)",
                },
            },
        },
    },
    # TODO(hallazgo-D, diagnostico-plan-agentes-multi-agente.md): this tool ranks by gross
    # margin % only -- it has no `sort_by` to rank by units sold or revenue, so it cannot
    # answer a literal "best sellers by units/revenue" question either. Out of scope for
    # this round -- do NOT add a `sort_by` param here without backend coordination. See
    # the escalation ticket in the Subagente 3 report for the two options under evaluation.
    {
        "name": "get_product_profitability",
        "description": "Calcula y consulta el ranking de rentabilidad y margen bruto % (((Ventas - Costos)/Ventas)*100) por producto o categoría. Acepta filtros de fecha para acotar el análisis a un periodo específico (mes, trimestre, año).",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "group_by": {
                    "type": "STRING",
                    "description": "Agrupación: 'product', 'category', 'brand', 'supplier'",
                    "enum": ["product", "category", "brand", "supplier"],
                },
                "limit": {
                    "type": "INTEGER",
                    "description": "Cantidad máxima de elementos en el ranking",
                },
                "date_from": {
                    "type": "STRING",
                    "description": "Fecha de inicio en formato YYYY-MM-DD (opcional). Usa cuando el usuario pida datos de un periodo concreto.",
                },
                "date_to": {
                    "type": "STRING",
                    "description": "Fecha de fin en formato YYYY-MM-DD (opcional). Usa cuando el usuario pida datos de un periodo concreto.",
                },
            },
        },
    },
    {
        "name": "get_funnel_and_cart_metrics",
        "description": "Obtiene métricas de conversión del funnel de ventas, tasa de abandono de carritos, productos más abandonados y ROI de cupones.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "timeframe": {
                    "type": "STRING",
                    "description": "Ventana temporal de análisis: '7d', '30d', '90d'",
                    "enum": ["7d", "30d", "90d"],
                },
            },
        },
    },
    {
        "name": "get_customer_reviews_summary",
        "description": "Obtiene el resumen de reseñas de clientes, rating promedio (1-5 estrellas), distribución de satisfacción y feedback crítico.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "product_id": {
                    "type": "STRING",
                    "description": "ID del producto para filtrar reseñas (opcional, por defecto 'all')",
                },
                "sentiment": {
                    "type": "STRING",
                    "description": "Filtro de sentimiento: 'all', 'positive', 'critical', 'negative'",
                    "enum": ["all", "positive", "critical", "negative"],
                },
            },
        },
    },
    {
        "name": "get_customer_segmentation",
        "description": "Consulta la segmentación de clientes mediante análisis RFM (VIPs, En Riesgo >60d, Nuevos <30d), Customer LTV y geolocalización.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "segment": {
                    "type": "STRING",
                    "description": "Segmento objetivo: 'all', 'vip', 'at_risk', 'new', 'churned'",
                    "enum": ["all", "vip", "at_risk", "new", "churned"],
                },
            },
        },
    },
    # NOTE: the old `semantic_catalog_search` declaration used to live here. It moved to
    # CATALOG_RAG_TOOL_DECLARATIONS below, where it gained pgvector semantics and hard
    # metadata filters: catalog retrieval is an e-commerce concern, not an analytics one.
    {
        "name": "execute_raw_sql_sandbox",
        "description": "Ejecuta de forma segura una consulta SQL de solo lectura (SELECT) en la base de datos analítica con límite de 50 filas.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "sql_query": {
                    "type": "STRING",
                    "description": "Consulta SQL (únicamente sentencias SELECT)",
                },
                "max_rows": {
                    "type": "INTEGER",
                    "description": "Límite de filas a retornar (máximo 50)",
                },
            },
            "required": ["sql_query"],
        },
    },
]

# Catalog retrieval tools (Fase 3 RAG). Exposed to the E-Commerce agent only: they are
# the model-driven counterpart of the eager catalog grounding, and none of them can
# reach the analytics SQL console.
CATALOG_RAG_TOOL_DECLARATIONS = [
    {
        "name": "semantic_catalog_search",
        "description": (
            "Busca productos en el catálogo por similitud semántica (significado, no solo coincidencia "
            "de palabras) combinada con filtros duros de metadatos (precio, categoría, marca, stock). "
            "Úsala cuando el usuario describa en lenguaje natural lo que quiere en lugar de nombrar un "
            "producto exacto. Reemplaza a la búsqueda por palabras clave: prefiérela siempre para "
            "intenciones difusas o aproximadas."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "query": {
                    "type": "STRING",
                    "description": "Texto tal como lo expresó el usuario, sin traducir ni resumir.",
                },
                "top_k": {
                    "type": "INTEGER",
                    "description": "Máximo de resultados. Default 8, máximo 20.",
                },
                "min_price": {
                    "type": "NUMBER",
                    "description": "Precio mínimo inclusive para acotar los resultados.",
                },
                "max_price": {
                    "type": "NUMBER",
                    "description": "Precio máximo inclusive para acotar los resultados.",
                },
                "category": {
                    "type": "STRING",
                    "description": "Nombre EXACTO de categoría, de list_catalog_facets. Nunca inventar uno.",
                },
                "brand": {
                    "type": "STRING",
                    "description": "Nombre EXACTO de marca, de list_catalog_facets.",
                },
                "in_stock_only": {
                    "type": "BOOLEAN",
                    "description": "Si es true, excluye productos sin stock. Default true.",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "check_stock_and_price",
        "description": (
            "Devuelve el stock y el precio EXACTOS y actuales leídos directamente de la base de datos en "
            "el momento de la consulta. Úsala SIEMPRE antes de confirmar disponibilidad o precio, incluso "
            "si esos datos ya aparecieron en semantic_catalog_search, porque pueden haber cambiado. "
            "Nunca inventes ni supongas stock o precio."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "item_ids": {
                    "type": "ARRAY",
                    "description": "Lista de IDs numéricos de productos a verificar.",
                    "items": {"type": "INTEGER"},
                },
                "slugs": {
                    "type": "ARRAY",
                    "description": "Lista de slugs de productos a verificar.",
                    "items": {"type": "STRING"},
                },
            },
            "required": [],
        },
    },
    {
        "name": "find_similar_products",
        "description": (
            "Dado un producto, devuelve otros semánticamente cercanos, ordenados por proximidad "
            "vectorial. Úsala para pedidos del tipo 'algo parecido a X' o para cross-sell una vez que "
            "el interés del cliente por un producto ya está confirmado."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "item_id": {
                    "type": "INTEGER",
                    "description": "ID numérico del producto de referencia.",
                },
                "top_k": {
                    "type": "INTEGER",
                    "description": "Default 5, máximo 15.",
                },
                "exclude_out_of_stock": {
                    "type": "BOOLEAN",
                    "description": "Default true.",
                },
            },
            "required": ["item_id"],
        },
    },
    {
        "name": "list_catalog_facets",
        "description": (
            "Devuelve los valores válidos de categoría y/o marca existentes en la base de datos. Úsala "
            "ANTES de filtrar semantic_catalog_search por categoría o marca, para no inventar nunca un "
            "nombre que no existe en la base de datos."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "facet": {
                    "type": "STRING",
                    "description": "Faceta a listar: 'category', 'brand' o 'both'.",
                    "enum": ["category", "brand", "both"],
                },
            },
            "required": ["facet"],
        },
    },
]

# Full catalogue of dispatchable tools. Note this is the union of what EXISTS, never of
# what a given agent is ALLOWED to call — see `execute_tool(allowed_tools=...)`.
ALL_TOOL_DECLARATIONS = [*ANALYTICS_TOOL_DECLARATIONS, *CATALOG_RAG_TOOL_DECLARATIONS]

# Human-readable Spanish labels surfaced to the end user as SSE progress events while a
# chained tool call is running (a warm chain takes 4-7s, a cold one far longer).
TOOL_PROGRESS_LABELS: dict[str, str] = {
    "query_sales_analytics": "Consultando métricas de ventas...",
    "get_inventory_health": "Revisando el estado del inventario...",
    "get_product_profitability": "Calculando márgenes y rentabilidad...",
    "get_funnel_and_cart_metrics": "Analizando el embudo de conversión...",
    "get_customer_reviews_summary": "Resumiendo reseñas de clientes...",
    "get_customer_segmentation": "Segmentando clientes (RFM)...",
    SQL_SANDBOX_TOOL_NAME: "Ejecutando consulta SQL de solo lectura...",
    "semantic_catalog_search": "Buscando en el catálogo...",
    "check_stock_and_price": "Verificando stock y precio...",
    "find_similar_products": "Buscando productos similares...",
    "list_catalog_facets": "Consultando categorías y marcas...",
}

DEFAULT_TOOL_PROGRESS_LABEL = "Consultando datos..."


def get_tool_label(tool_name: str) -> str:
    """Returns the Spanish progress label displayed while a tool runs.

    Args:
        tool_name: The declared tool name identifier.

    Returns:
        The mapped label, or a generic default for unmapped/unknown tool names.
    """
    return TOOL_PROGRESS_LABELS.get(tool_name, DEFAULT_TOOL_PROGRESS_LABEL)


def _coerce_int_list(raw: Any) -> list[int]:
    """Coerces a model-supplied sequence into a list of ints, dropping bad entries.

    Gemini frequently emits numeric arguments as strings ("12" instead of 12); a single
    non-numeric entry must not abort the whole verification call.

    Args:
        raw: The raw argument value emitted by the model.

    Returns:
        A list of successfully coerced integers (empty when nothing was coercible).
    """
    if raw is None:
        return []
    if not isinstance(raw, (list, tuple, set)):
        raw = [raw]

    coerced: list[int] = []
    for entry in raw:
        try:
            coerced.append(int(entry))
        except (TypeError, ValueError):
            logger.debug("Ignoring non-coercible item id from model: %r", entry)
    return coerced


async def execute_tool(
    tool_name: str,
    tool_args: dict[str, Any],
    user_token: Optional[str] = None,
    allowed_tools: Optional[Iterable[str]] = None,
) -> dict[str, Any]:
    """Dispatches a tool call invocation to the corresponding Django API service method.

    Args:
        tool_name: The declared tool name identifier.
        tool_args: Dictionary of arguments passed by the model.
        user_token: Optional JWT token for authenticated operations.
        allowed_tools: Optional server-side allowlist. When provided, any tool outside
            it is refused WITHOUT dispatching. This is the second authorization layer:
            the first is simply never showing the schema to the model, but a
            hallucinated or prompt-injected tool name would bypass that one alone.

    Returns:
        Dictionary containing the tool execution result.
    """
    if allowed_tools is not None and tool_name not in set(allowed_tools):
        logger.warning(
            "Blocked tool '%s': not in the allowlist for this agent.", tool_name
        )
        return {
            "status": "error",
            "blocked": True,
            "error": f"La herramienta '{tool_name}' no está habilitada para este agente.",
        }

    django_service = get_django_api_service()
    logger.info("Executing tool '%s' with args: %s", tool_name, tool_args)

    try:
        if tool_name == "query_sales_analytics":
            return await django_service.query_sales_analytics(
                date_from=tool_args.get("date_from"),
                date_to=tool_args.get("date_to"),
                dimension=tool_args.get("dimension", "category"),
                user_token=user_token,
            )

        elif tool_name == "get_inventory_health":
            return await django_service.get_inventory_health(
                status_filter=tool_args.get("status_filter", "all"),
                limit=int(tool_args.get("limit", 10)),
                user_token=user_token,
            )

        elif tool_name == "get_product_profitability":
            return await django_service.get_product_profitability(
                group_by=tool_args.get("group_by", "product"),
                limit=int(tool_args.get("limit", 10)),
                date_from=tool_args.get("date_from"),
                date_to=tool_args.get("date_to"),
                user_token=user_token,
            )

        elif tool_name == "get_funnel_and_cart_metrics":
            return await django_service.get_funnel_and_cart_metrics(
                timeframe=tool_args.get("timeframe", "30d"),
                user_token=user_token,
            )

        elif tool_name == "get_customer_reviews_summary":
            return await django_service.get_customer_reviews_summary(
                product_id=tool_args.get("product_id"),
                sentiment=tool_args.get("sentiment", "all"),
                user_token=user_token,
            )

        elif tool_name == "get_customer_segmentation":
            return await django_service.get_customer_segmentation(
                segment=tool_args.get("segment", "all"),
                user_token=user_token,
            )

        elif tool_name == "semantic_catalog_search":
            # Never raises: degrades to the legacy lexical engine and tags the payload.
            return await semantic_catalog_search_with_fallback(
                query=tool_args.get("query", ""),
                top_k=max(1, min(int(tool_args.get("top_k") or 8), 20)),
                min_price=tool_args.get("min_price"),
                max_price=tool_args.get("max_price"),
                category=tool_args.get("category"),
                brand=tool_args.get("brand"),
                in_stock_only=bool(tool_args.get("in_stock_only", True)),
            )

        elif tool_name == "check_stock_and_price":
            item_ids = _coerce_int_list(tool_args.get("item_ids"))
            raw_slugs = tool_args.get("slugs") or []
            if not isinstance(raw_slugs, (list, tuple, set)):
                raw_slugs = [raw_slugs]
            slugs = [str(slug) for slug in raw_slugs if str(slug).strip()]
            return await django_service.verify_items(
                item_ids=item_ids or None,
                slugs=slugs or None,
            )

        elif tool_name == "find_similar_products":
            return await find_similar_products_with_fallback(
                item_id=int(tool_args.get("item_id", 0)),
                top_k=max(1, min(int(tool_args.get("top_k") or 5), 15)),
                exclude_out_of_stock=bool(tool_args.get("exclude_out_of_stock", True)),
            )

        elif tool_name == "list_catalog_facets":
            return await django_service.get_catalog_facets(facet=tool_args.get("facet", "both"))

        elif tool_name == SQL_SANDBOX_TOOL_NAME:
            return await django_service.execute_raw_sql_sandbox(
                sql_query=tool_args.get("sql_query", ""),
                max_rows=int(tool_args.get("max_rows", 50)),
                user_token=user_token,
            )

        else:
            logger.warning("Unknown tool '%s' requested.", tool_name)
            return {"status": "error", "error": f"Tool '{tool_name}' is not recognized."}

    except Exception as exc:
        logger.error("Error executing tool '%s': %s", tool_name, exc, exc_info=True)
        return {"status": "error", "error": f"Tool execution failed: {str(exc)}"}
